import os
import json
import sys
import numpy as np
from sklearn import metrics
import logging
import copy

from mmod.io_utils import read_to_buffer
from mmod.utils import open_file


def rect_area(rc):
    return (rc[2] - rc[0] + 1) * (rc[3] - rc[1] + 1)


def intersection(rc1, rc2):
    rc_inter = [max(rc1[0], rc2[0]), max(rc1[1], rc2[1]), min(rc1[2], rc2[2]), min(rc1[3], rc2[3])]
    iw = rc_inter[2] - rc_inter[0] + 1
    ih = rc_inter[3] - rc_inter[1] + 1
    return float(iw) * ih if (iw > 0 and ih > 0) else 0


# calculate the Jaccard similarity between two rectangles
def iou(rc1, rc2):
    if len(rc1) == 0 or len(rc2) == 0:
        return 0
    rc_inter = [max(rc1[0], rc2[0]), max(rc1[1], rc2[1]), min(rc1[2], rc2[2]), min(rc1[3], rc2[3])]
    iw = rc_inter[2] - rc_inter[0] + 1
    ih = rc_inter[3] - rc_inter[1] + 1
    return (float(iw)) * ih / (rect_area(rc1) + rect_area(rc2) - iw * ih) if (iw > 0 and ih > 0) else 0


def evaluate_(c_detects, c_truths, ovthresh):
    """
    For each detection in a class, check whether it hits a ground truth box or not
    Return: (a list of confs, a list of hit or miss, number of ground truth boxes)
    """
    npos = 0
    for img_id in c_truths:
        npos += len([difficulty_gtbox for difficulty_gtbox in c_truths[img_id] if difficulty_gtbox[0] == 0])

    y_trues = []
    y_scores = []

    dettag = set()

    for i in range(len(c_detects)):
        det = c_detects[i]
        y_true = 0
        img_id = det[0]
        conf = det[1]
        bbox = det[2]
        if img_id in c_truths:
            if ovthresh < 0 and (bbox is None or len(bbox) == 0):
                if img_id not in dettag:
                    y_true = 1
                    dettag.add(img_id)
            else:
                # get overlaps with truth rectangles
                overlaps = np.array([iou(bbox, gtbox[1]) for gtbox in c_truths[img_id]])
                bbox_idx_max = np.argmax(overlaps)
                if overlaps[bbox_idx_max] > ovthresh:
                    # if a detection hits a difficult gt_box, skip this detection
                    if c_truths[img_id][bbox_idx_max][0] != 0:
                        continue

                    if (img_id, bbox_idx_max) not in dettag:
                        y_true = 1
                        dettag.add((img_id, bbox_idx_max))

        y_trues += [y_true]
        y_scores += [conf]
    return np.array(y_scores), np.array(y_trues), npos


# split the file path into (directory, basename, ext)
def splitpath(filepath):
    (path, fname) = os.path.split(filepath)
    (basename, ext) = os.path.splitext(fname)
    return path, basename, ext


def eval_one(truths, detresults, ovthresh=-1, confs=None, label_to_keys=None):
    if confs is None:
        confs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_scores = []
    y_trues = []
    npos = 0
    class_thresh = dict()
    apdict = dict()
    for label in sorted(truths.keys()):
        if label not in detresults:
            apdict[label] = 0
            continue
        c_detects = detresults[label]        # detection results for current class
        c_truths = truths[label]             # truths for current class
        if label_to_keys is not None:
            valid_keys = label_to_keys.get(label, [])
            c_truths = {key: c_truths[key] for key in c_truths if key in valid_keys}
            c_detects = [(key, conf, bbox) for key, conf, bbox in c_detects if key in valid_keys]
        (c_y_scores, c_y_trues, c_npos) = evaluate_(c_detects, c_truths, ovthresh)
        if confs and np.sum(c_y_trues):
            precision, recall, thresholds = metrics.precision_recall_curve(c_y_trues, c_y_scores)
            for conf in confs:
                indices, = np.where((precision > conf) & (recall > 0.0))
                if len(indices) == 0:
                    continue
                if label not in class_thresh:
                    class_thresh[label] = {}
                class_thresh[label].update({
                    conf: (thresholds[indices[0]], recall[indices[0]])
                })
        if len(c_detects) > 0:
            c_true_sum = np.sum(c_y_trues)
            ap = metrics.average_precision_score(c_y_trues, c_y_scores) * c_true_sum/c_npos if c_true_sum > 0 else 0
            y_scores += [c_y_scores]
            y_trues += [c_y_trues]
            apdict[label] = ap
        else:
            apdict[label] = 0
        npos += c_npos
    m_ap = sum(apdict.values()) / (1 if len(truths) == 0 else len(truths))
    y_trues = np.hstack(y_trues) if len(y_trues) != 0 else np.array([0])
    y_scores = np.hstack(y_scores) if len(y_scores) != 0 else np.array([0])
    coverage_ratio = float(np.sum(y_trues))/npos if npos != 0 else 0
    if np.sum(y_trues) == 0:
        # in this case, metrics.precision_recall_curve will crash
        precision = np.asarray([0.])
        recall = np.asarray([0.])
        thresholds = np.asarray([0.])
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_trues, y_scores)
    precision = list(precision)
    thresholds = list(thresholds)
    if len(thresholds) < len(precision):
        thresholds += [thresholds[-1]]
    recall *= coverage_ratio
    recall = list(recall)
    return {
        'class_ap': apdict,
        'class_thresh': class_thresh,
        'map': m_ap,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'npos': npos,
        'coverage_ratio': coverage_ratio
    }


def get_pr(report, thresh):
    idx = np.where(np.array(report['precision']) > thresh)
    if len(idx) == 0:
        return 0, 0, 0
    recall_ = np.array(report['recall'])[idx]
    if len(recall_) == 0:
        return 0, 0, 0
    maxid = np.argmax(np.array(recall_))
    maxid = idx[0][maxid]
    return report['thresholds'][maxid], report['precision'][maxid], report['recall'][maxid]


def print_pr(report, thresh):
    th, prec, rec = get_pr(report, thresh)
    print("\t%9.6f\t%9.6f\t%9.6f" % (th, prec, rec))


def get_report(exp, dets, ovths):
    """Evaluate Detection
    :param exp: The experiment related to predictions
    :type exp: mmod.experiment.Experiment
    :param dets: detection results
    :param ovths: overlap thresholds
    """
    truths = exp.imdb.all_truths()
    assert truths, "no truth is given"

    logging.info('Generating report for {}'.format(exp))
    truths_list = {'overall': truths}
    reports = dict()
    for part in truths_list:
        reports[part] = dict()
        for ov_th in ovths:
            reports[part][ov_th] = eval_one(truths_list[part], dets, ov_th)
    return reports  # return the overal reports


def print_reports(reports, precths=None, report_file_table=None):
    if precths is None:
        precths = [0.8, 0.9, 0.95]
    headings = ['IOU', 'MAP']
    for precth in precths:
        headings += ['Th@%g' % precth, 'P%g' % precth, 'R@%g' % precth]
    with open(report_file_table, 'w') if report_file_table else open_file(sys.stdout) as fp:
        for key in reports:
            table = []
            mv_report = reports[key]
            for ov_th in sorted(mv_report.keys()):
                report = mv_report[ov_th]
                data = [ov_th, report['map']]
                for precth in precths:
                    pr = get_pr(report, precth)
                    data += list(pr)
                data = tuple(['{}'.format(round(x, 4)) for x in data])
                table += [data]
            note = ('Results on %s objects (%d)' % (key, mv_report[mv_report.keys()[0]]['npos']))
            logging.info(note)
            fp.write(note + '\n')
            line = '\t'.join(headings) + '\n'
            logging.info(line)
            fp.write(line)
            for data in table:
                line = '\t'.join(data) + '\n'
                logging.info(line)
                fp.write(line)


def lift_detects(detresults, label_tree):
    logging.info("Lifting the detections for hierarcy ")
    result = {}
    for label in detresults:
        dets = detresults[label]
        all_label = [label]
        nodes = label_tree.root.search_nodes(name=label)
        assert len(nodes) == 1
        node = nodes[0]
        for n in node.get_ancestors()[: -1]:
            all_label.append(n.name)
        for l in all_label:
            if l not in result:
                result[l] = copy.deepcopy(dets)
            else:
                result[l].extend(dets)
    return result


def lift_truths(truths, label_tree):
    logging.info("Lifting the truth for hierarcy ")
    result = {}
    for label in truths:
        imid_to_rects = truths[label]
        all_label = [label]
        nodes = label_tree.root.search_nodes(name=label)
        assert len(nodes) == 1
        node = nodes[0]
        for n in node.get_ancestors()[: -1]:
            all_label.append(n.name)
        logging.debug('->{}'.format(','.join(all_label)))
        for l in all_label:
            if l not in result:
                result[l] = copy.deepcopy(imid_to_rects)
            else:
                r = result[l]
                for imid in imid_to_rects:
                    rects = imid_to_rects[imid]
                    if imid in r:
                        r[imid].extend(rects)
                    else:
                        r[imid] = rects
    return result


def deteval(exp,
            name='',
            precths=None, ovthresh=None):
    """Evaluate Detection
    :param exp: The experiment related to predictions
    :type exp: mmod.experiment.Experiment
    :param name: name to override experiment name
    :param precths: Percentages
    :param ovthresh: overlap threshold
    """
    if precths is None:
        precths = [0.8, 0.9, 0.95]
    if ovthresh is None:
        ovthresh = [-1.0, 0.3, 0.5]

    outtsv_file = exp.predict_path

    logging.info("Evaluating {} in {}".format(outtsv_file, exp))

    report_dir, fbase, _ = splitpath(outtsv_file)

    # save the evaluation result to the report file, which can be used as baseline
    exp_name = name if name != "" else fbase
    report_name = exp_name if report_dir == '' else '/'.join([report_dir, exp_name])
    report_file = report_name + ".report"
    brief_report_file = report_file + '.table'

    if os.path.isfile(report_file):
        if os.path.isfile(brief_report_file):
            logging.info('skip to evaluate since the last report file exists: {}'.format(brief_report_file))
            return report_file, None
        logging.info("Read report from: {}".format(report_file))
        reports = json.loads(read_to_buffer(report_file))
    else:
        detresults = exp.load_detections()

        # brief report on different object size
        reports = get_report(exp, detresults, ovthresh)

        # detail report with p-r curve
        logging.info('Saving the report: {}'.format(report_file))
        with open(report_file, "w") as fout:
            fout.write(json.dumps(reports, separators=(',', ':'), sort_keys=True))

    print_reports(reports, precths, brief_report_file)

    return report_file, reports
