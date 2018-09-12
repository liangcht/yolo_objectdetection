import sys
import numpy as np
sys.path.insert(0, "d:/development/Caffe/python")
import caffe
import logging


def init_logging():
    np.seterr(all='raise')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s',
                        datefmt='%m-%d %H:%M:%S',
    )


def convert_layout(o_last_conv, with_tree=False):
    n_last_conv = np.zeros(o_last_conv.shape)
    num_anchor = o_last_conv.shape[1] / 25
    for b in range(o_last_conv.shape[0]):
        for a in range(num_anchor):
            # x
            n_last_conv[b, a, :, :] = o_last_conv[b, a * 25 + 0, :, :]
            # y
            n_last_conv[b, num_anchor + a, :, :] = o_last_conv[b, a * 25 + 1, :, :]
            # w
            n_last_conv[b, 2 * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 2, :, :]
            # h
            n_last_conv[b, 3 * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 3, :, :]
            # o
            n_last_conv[b, 4 * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 4, :, :]
            # cls
            if with_tree:
                for c in range(20):
                    n_last_conv[b, (5 + c) * num_anchor + a, :, :] = o_last_conv[b, a * 25 + 5 + c, :, :]
            else:
                n_last_conv[b, 5 * num_anchor + a * 20: 5 * num_anchor + a * 20 +
                        20, :, :] = o_last_conv[b, a * 25 + 5 : a * 25 + 25, :, :]
    return n_last_conv


def check_yolo_test_full_gpu():
    new_layer_proto = './aux_data/new_yolo_test.prototxt'
    old_layer_proto = './aux_data/old_yolo_test.prototxt'
    caffe.set_mode_gpu()
    o = caffe.Net(old_layer_proto, caffe.TEST)
    n = caffe.Net(new_layer_proto, caffe.TEST)
    #np.random.seed(779)
    batch_size = 1
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    im_info = [200, 200]
    
    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['im_info'].data[...] = im_info
    o.forward()
    o_prob = o.blobs['prob'].data[0]
    o_bbox = o.blobs['bbox'].data[0]

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['im_info'].data[...] = im_info
    n.forward()
    n_bbox = n.blobs['bbox'].data[0]
    n_bbox = n_bbox.reshape(-1, n_bbox.shape[-1])
    n_prob = n.blobs['prob'].data[0]
    n_prob = n_prob.reshape(-1, n_prob.shape[-1])

    x = np.abs(n_bbox[:] - o_bbox[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_bbox.shape))
    logging.info(n_bbox.reshape(-1)[idx])
    logging.info(o_bbox.reshape(-1)[idx])

    y = np.abs(n_prob[:] - o_prob[:]).reshape(-1)
    idx = np.argmax(y)
    logging.info(idx)
    logging.info(np.sum(y))
    logging.info(np.unravel_index(idx, n_prob.shape))
    logging.info(n_prob.reshape(-1)[idx])
    logging.info(o_prob.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


def check_yolo_full_gpu():
    new_layer_proto = './aux_data/new_layers.prototxt'
    old_layer_proto = './aux_data/old_layer.prototxt'
    caffe.set_mode_gpu()
    n = caffe.Net(new_layer_proto, caffe.TRAIN)
    o = caffe.Net(old_layer_proto, caffe.TRAIN)
    #np.random.seed(779)
    batch_size = 6
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    label = np.zeros((6, 150))
    for i in range(batch_size):
        for j in range(15):
            label[i, 5 * j + 0] = np.random.rand()
            label[i, 5 * j + 1] = np.random.rand()
            label[i, 5 * j + 2] = np.random.rand()
            label[i, 5 * j + 3] = np.random.rand()
            label[i, 5 * j + 4] = 5
    
    iter_number = 158000
    #iter_number = 0

    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['label'].data[...] = label
    o.params['region_loss'][0].data[0] = iter_number
    o.params['region_loss']
    o.forward()
    o.backward()
    o_diff= o.blobs['last_conv'].diff

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['label'].data[...] = label
    n.params['region_target'][0].data[0] = iter_number
    n.forward()
    n.backward()
    n_diff = n.blobs['last_conv'].diff

    on_diff = convert_layout(o_diff)
    x = np.abs(n_diff[:] - on_diff[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_diff.shape))
    logging.info(n_diff.reshape(-1)[idx])
    logging.info(on_diff.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


def check_yolo_tree_test_full_gpu():
    new_layer_proto = './aux_data/new_yolo_tree_test.prototxt'
    old_layer_proto = './aux_data/old_yolo_tree_test.prototxt'
    caffe.set_mode_gpu()
    o = caffe.Net(old_layer_proto, caffe.TEST)
    n = caffe.Net(new_layer_proto, caffe.TEST)
    #np.random.seed(779)
    batch_size = 1
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv, True) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    im_info = [200, 200]
    
    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['im_info'].data[...] = im_info
    o.forward()
    o_prob = o.blobs['prob'].data[0]
    o_bbox = o.blobs['bbox'].data[0]

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['im_info'].data[...] = im_info
    n.forward()
    n_bbox = n.blobs['bbox'].data[0]
    n_bbox = n_bbox.reshape(-1, n_bbox.shape[-1])
    n_prob = n.blobs['prob'].data[0]
    n_prob = n_prob.reshape(-1, n_prob.shape[-1])

    x = np.abs(n_bbox[:] - o_bbox[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_bbox.shape))
    logging.info(n_bbox.reshape(-1)[idx])
    logging.info(o_bbox.reshape(-1)[idx])

    y = np.abs(n_prob[:] - o_prob[:]).reshape(-1)
    idx = np.argmax(y)
    logging.info(idx)
    logging.info(np.sum(y))
    logging.info(np.unravel_index(idx, n_prob.shape))
    logging.info(n_prob.reshape(-1)[idx])
    logging.info(o_prob.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


def check_yolo_tree_classnms_test_full_gpu():
    new_layer_proto = './aux_data/new_yolo_tree_classnms_test.prototxt'
    old_layer_proto = './aux_data/old_yolo_tree_classnms_test.prototxt'
    caffe.set_mode_gpu()
    o = caffe.Net(old_layer_proto, caffe.TEST)
    n = caffe.Net(new_layer_proto, caffe.TEST)
    #np.random.seed(779)
    batch_size = 1
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv, True) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    im_info = [200, 200]
    
    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['im_info'].data[...] = im_info
    o.forward()
    o_prob = o.blobs['prob'].data[0]
    o_bbox = o.blobs['bbox'].data[0]

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['im_info'].data[...] = im_info
    n.forward()
    n_bbox = n.blobs['bbox'].data[0]
    n_bbox = n_bbox.reshape(-1, n_bbox.shape[-1])
    n_prob = n.blobs['prob'].data[0]
    n_prob = n_prob.reshape(-1, n_prob.shape[-1])

    x = np.abs(n_bbox[:] - o_bbox[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_bbox.shape))
    logging.info(n_bbox.reshape(-1)[idx])
    logging.info(o_bbox.reshape(-1)[idx])

    y = np.abs(n_prob[:] - o_prob[:]).reshape(-1)
    idx = np.argmax(y)
    logging.info(idx)
    logging.info(np.sum(y))
    logging.info(np.unravel_index(idx, n_prob.shape))
    logging.info(n_prob.reshape(-1)[idx])
    logging.info(o_prob.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


def check_yolo_tree_full_gpu():
    new_layer_proto = './aux_data/new_layers_tree.prototxt'
    old_layer_proto = './aux_data/old_layer_tree.prototxt'
    caffe.set_mode_gpu()
    n = caffe.Net(new_layer_proto, caffe.TRAIN)
    o = caffe.Net(old_layer_proto, caffe.TRAIN)
    #np.random.seed(779)
    batch_size_bb = 4
    batch_size_nobb = 2
    batch_size = batch_size_bb + batch_size_nobb
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv, True) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    label_bb = np.zeros((4, 150))
    for i in range(batch_size_bb):
        for j in range(15):
            label_bb[i, 5 * j + 0] = np.random.rand()
            label_bb[i, 5 * j + 1] = np.random.rand()
            label_bb[i, 5 * j + 2] = np.random.rand()
            label_bb[i, 5 * j + 3] = np.random.rand()
            label_bb[i, 5 * j + 4] = 5
    label_nobb = np.zeros((2, 150))
    for i in range(batch_size_nobb):
        for j in range(15):
            label_nobb[i, 5 * j + 0] = 999999
            label_nobb[i, 5 * j + 1] = 999999
            label_nobb[i, 5 * j + 2] = 999999
            label_nobb[i, 5 * j + 3] = 999999
            label_nobb[i, 5 * j + 4] = 5
    
    iter_number = 158000
    #iter_number = 0

    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['label_bb'].data[...] = label_bb
    o.blobs['label_nobb'].data[...] = label_nobb
    o.params['region_loss'][0].data[0] = iter_number
    o.params['region_loss']
    o.forward()
    o.backward()
    o_diff= o.blobs['last_conv'].diff

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['label_bb'].data[...] = label_bb
    n.blobs['label_nobb'].data[...] = label_nobb
    n.params['region_target'][0].data[0] = iter_number
    n.forward()
    n.backward()
    n_diff = n.blobs['last_conv'].diff

    on_diff = convert_layout(o_diff, True)
    x = np.abs(n_diff[:] - on_diff[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_diff.shape))
    logging.info(n_diff.reshape(-1)[idx])
    logging.info(on_diff.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


def check_yolo_tree_bb_full_gpu():
    new_layer_proto = './aux_data/new_layers_tree_bb.prototxt'
    old_layer_proto = './aux_data/old_layer_tree_single_label.prototxt'
    caffe.set_mode_gpu()
    n = caffe.Net(new_layer_proto, caffe.TRAIN)
    o = caffe.Net(old_layer_proto, caffe.TRAIN)
    #np.random.seed(779)
    batch_size = 6
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv, True) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    label = np.zeros((6, 150))
    for i in range(batch_size):
        for j in range(15):
            label[i, 5 * j + 0] = np.random.rand()
            label[i, 5 * j + 1] = np.random.rand()
            label[i, 5 * j + 2] = np.random.rand()
            label[i, 5 * j + 3] = np.random.rand()
            label[i, 5 * j + 4] = 5
    
    iter_number = 158000
    #iter_number = 0

    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['label'].data[...] = label
    o.params['region_loss'][0].data[0] = iter_number
    o.params['region_loss']
    o.forward()
    o.backward()
    o_diff= o.blobs['last_conv'].diff

    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['label'].data[...] = label
    n.params['region_target'][0].data[0] = iter_number
    n.forward()
    n.backward()
    n_diff = n.blobs['last_conv'].diff

    on_diff = convert_layout(o_diff, True)
    x = np.abs(n_diff[:] - on_diff[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_diff.shape))
    logging.info(n_diff.reshape(-1)[idx])
    logging.info(on_diff.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


def check_yolo_tree_nobb_full_gpu():
    new_layer_proto = './aux_data/new_layers_tree_nobb.prototxt'
    old_layer_proto = './aux_data/old_layer_tree_single_label.prototxt'
    caffe.set_mode_gpu()
    n = caffe.Net(new_layer_proto, caffe.TRAIN)
    o = caffe.Net(old_layer_proto, caffe.TRAIN)
    #np.random.seed(779)
    batch_size = 6
    num_anchor = 5
    o_last_conv = np.random.rand(batch_size, 125, 13, 13)
    n_last_conv = convert_layout(o_last_conv, True) 

    logging.info(np.mean(o_last_conv[:]))
    logging.info(np.mean(n_last_conv[:]))

    label = np.zeros((6, 150))
    for i in range(batch_size):
        for j in range(15):
            label[i, 5 * j + 0] = 999999
            label[i, 5 * j + 1] = 999999
            label[i, 5 * j + 2] = 999999
            label[i, 5 * j + 3] = 999999
            label[i, 5 * j + 4] = 5
    
    iter_number = 158000
    #iter_number = 0

    o.blobs['last_conv'].data[...] = o_last_conv
    o.blobs['label'].data[...] = label
    o.params['region_loss'][0].data[0] = iter_number
    o.params['region_loss']
    o.forward()
    o.backward()
    o_diff= o.blobs['last_conv'].diff
    n.blobs['last_conv'].data[...] = n_last_conv
    n.blobs['label'].data[...] = label
    n.forward()
    n.backward()
    n_diff = n.blobs['last_conv'].diff

    on_diff = convert_layout(o_diff, True)
    x = np.abs(n_diff[:] - on_diff[:]).reshape(-1)
    idx = np.argmax(x)
    logging.info(idx)
    logging.info(np.sum(x))
    logging.info(np.unravel_index(idx, n_diff.shape))
    logging.info(n_diff.reshape(-1)[idx])
    logging.info(on_diff.reshape(-1)[idx])
    #import ipdb;ipdb.set_trace()


if __name__ == '__main__':
    init_logging()
    #remove_bb_train_test()
    #remove_bb_train_test2()
    #yolo9000_coco50K()
    #yolo9000()
    #study_loss_per_cat()
    #smaller_network_input_size()

    #check_yolo_full_gpu()
    #check_yolo_test_full_gpu()

    #check_yolo_tree_full_gpu()
    #check_yolo_tree_bb_full_gpu()
    #check_yolo_tree_nobb_full_gpu()
    #check_yolo_tree_test_full_gpu()
    #check_yolo_tree_classnms_test_full_gpu()

    #low_shot_checking()
    #compare()

    #compare_log_for_multibin()
    #study_target()
    #test_()
    #visualize_multibin()
    #visualize_multibin2()

    #test_demo()
    #all_flops()
    #mobile_net()
    #test_yolo9000_on_imagenet()
    #cifar()
    #classification_task()
    #print_label_order()
    #officev2_1()
    #officev2_11()
    #yolo_study()
    #test_devonc()
    #for_taxonomy()

