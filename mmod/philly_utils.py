import os
import os.path as op
import re
import logging

from mmod.utils import makedirs


def _vc_home():
    """Find philly's VC home in scratch space
    :rtype: str
    """
    home = os.environ.get('PHILLY_VC_NFS_DIRECTORY', os.environ.get('PHILLY_VC_DIRECTORY'))
    if not home:
        home = op.expanduser('~').replace("\\", "/")
        home = '/'.join(home.split('/')[:5])
    return home


_VC_HOME = _vc_home()


def vc_name():
    """Find philly's VC name
    :rtype: str
    """
    name = os.environ.get('PHILLY_VC')
    if name:
        return name
    name = op.basename(_VC_HOME)
    if name:
        return name
    return op.basename(op.dirname(_VC_HOME))


_VC_NAME = vc_name()


def _vc_hdfs_base():
    base = os.environ.get("PHILLY_DATA_DIRECTORY") or os.environ.get("PHILLY_HDFS_PREFIX")
    if base:
        return base
    for base in ["/hdfs", "/home"]:
        if op.isdir(base):
            return base
    return '/'.join(_VC_HOME.split('/')[:-1])


def vc_hdfs_root():
    """Find the HDFS root of the VC
    :rtype: str
    """
    path = os.environ.get('PHILLY_VC_HDFS_DIRECTORY')
    if path:
        return path
    path = op.join(os.environ.get('PHILLY_HDFS_PREFIX', _vc_hdfs_base()), _VC_NAME)
    return path


_VC_HDFS_ROOT = vc_hdfs_root()


def expand_vc_user(path):
    """Expand ~ to VC's home
    :param path: the path to expand VC user
    :type path: str
    :return:/var/storage/shared/$VC_NAME
    :rtype: str
    """
    if path.startswith('~'):
        path = op.abspath(op.join(_VC_HOME, '.' + path[1:]))
    elif path.startswith('#'):
        path = op.abspath(op.join(_VC_HDFS_ROOT, '.' + path[1:]))

    return path


def abspath(path, roots=None):
    """Expand ~ to VC's home and resolve relative paths to absolute paths
    :param path: the path to resolve
    :type path: str
    :param roots: CWD roots to resolve relative paths to them
    :type roots: list
    """
    assert path, "{} is invalid path".format(path)
    path = expand_vc_user(path)
    if op.isabs(path):
        return path.replace("\\", "/")
    if not roots:
        roots = ["~"]
    roots = [expand_vc_user(root) for root in roots]
    for root in roots:
        resolved = op.abspath(op.join(root, path))
        if op.isfile(resolved) or op.isdir(resolved):
            return resolved.replace("\\", "/")
    # return assuming the first root (even though it does not exist)
    return op.abspath(op.join(roots[0], path)).replace("\\", "/")


def mirror_paths(paths):
    """Return absolute and mirror paths
    :param paths: list of paths moved/mirrored for philly
    :return: paths: absolute paths (resolved)
             relpaths: relative paths (relative to the vc root)
    """
    # find absolute paths to input solvers (input solvers with relative path assuemd to be relative to VC sctarch home)
    paths = [abspath(path, roots=['~', _VC_HDFS_ROOT]) for path in paths]

    relpaths = []
    for idx, path in enumerate(paths):
        if op.normpath(op.commonprefix([path, _VC_HOME])) == op.normpath(_VC_HOME):
            # if path is in VC root, keep structure relative to that root
            relpath = op.relpath(path, _VC_HOME)
        elif op.normpath(op.commonprefix([path, _VC_HDFS_ROOT])) == op.normpath(_VC_HDFS_ROOT):
            # if path is in VC HDFS root, keep structure relative to that root
            relpath = op.relpath(path, _VC_HDFS_ROOT)
        elif len(paths):
            relpath = op.join(str(idx), op.basename(path))
        else:
            relpath = op.basename(path)
        relpaths.append(relpath)
    return paths, relpaths


def attempt():
    """Philly attempt id
    :rtype: int
    """
    attempt_id = os.environ.get('OVERRIDE_ATTEMPT_ID') or os.environ.get('PHILLY_ATTEMPT_ID')
    if attempt_id:
        return int(attempt_id)
    log_base = get_log_parent()
    last_attempt = 0
    for attempt_id in os.listdir(log_base):
        if not op.isdir(op.join(log_base, attempt_id)):
            # ignore non-directories
            continue
        try:
            attempt_id = int(attempt_id)
        except ValueError:
            continue
        if attempt_id > last_attempt:
            last_attempt = attempt_id
    return last_attempt


def iterate_log_dir(log_parent=None):
    """Iterate log directories
    :param log_parent: log directory parent
    """
    if log_parent:
        last_attempt = 1
        logs = []
        while True:
            path = op.join(log_parent, "{}".format(last_attempt))
            if not op.isdir(path):
                break
            logs.append(path)
            last_attempt += 1
        for path in reversed(logs):
            yield path
        return
    log_base = get_log_parent()
    last_attempt = attempt()
    while last_attempt:
        yield op.join(log_base, "{}".format(last_attempt))
        last_attempt -= 1


def last_log_dir():
    """Get the last log directory of this job
    """
    log_dir = os.environ.get('OVERRIDE_LOG_DIRECTORY') or os.environ.get('PHILLY_LOG_DIRECTORY')
    if log_dir:
        return log_dir
    return op.join(get_log_parent(), "{}".format(attempt()))


def job_id(path=None):
    """Get the philly job ID (from a path)
    :param path:Path to seach for app id
    :rtype: str
    """
    if path is None:
        return os.environ.get('OVERRIDE_JOB_ID') or os.environ.get('PHILLY_JOB_ID') or job_id(op.expanduser('~'))
    m = re.search('/(?P<app_id>application_[\d_]+)[/\w]*$', path)
    if m:
        return m.group('app_id')
    logging.error("Job ID could not be detected")
    return ''


def set_job_id(expid):
    """Set job ID if experiment is done locally
    :param expid: Experiment ID
    :type expid: str
    """
    jid = job_id()
    if not expid:
        assert jid, ("Could not detect the job ID (perhaps not on Philly):"
                     " Provide Full experiment ID with:"
                     " a) --expid switch or "
                     " b) export OVERRIDE_JOB_ID environment variable.")
        return
    if not jid:
        assert expid
        jid = job_id(expid) or expid
    assert jid
    os.environ['OVERRIDE_JOB_ID'] = jid
    makedirs(get_log_parent(), exist_ok=True)
    new_attempt = attempt() + 1
    logging.info("Experiment: {} Attempt: {}".format(jid, new_attempt))
    os.environ['OVERRIDE_ATTEMPT_ID'] = "{}".format(new_attempt)
    os.environ['OVERRIDE_MODEL_DIRECTORY'] = ""  # model/output directory will depend on each solver
    makedirs(last_log_dir(), exist_ok=True)


def get_log_parent(path=None):
    """Find the parent directory of the logs
    :param path: a path from the job
    :rtype: str
    """
    jid = job_id(path)
    if not jid:
        return op.expanduser("~/logs")  # last resort
    return abspath(op.join('sys', 'jobs', jid, 'logs'), roots=['~'])


def get_model_path(path=None):
    """Find the default location to output/models
    """
    if not path:
        model_path = os.environ.get('OVERRIDE_MODEL_DIRECTORY')
        if model_path is not None:
            return model_path
        model_path = os.environ.get('PHILLY_MODEL_DIRECTORY')
        if model_path:
            return model_path
    return abspath(op.join('sys', 'jobs', job_id(path), 'models'), roots=[vc_hdfs_root()])


def get_arg(arg, cast=None):
    """Clean up arg
    :param arg: argument from command line
    :param cast: type of the argument to cast to
    """
    if arg and arg.lower().strip() == 'none':
        arg = None
    if arg and cast:
        arg = cast(arg)
    return arg


def get_master_machine():
    mpi_host_file = op.expanduser('~/mpi-hosts')
    with open(mpi_host_file, 'r') as f:
        master_name = f.readline().strip()
    return master_name


def get_master_ip(master_name=None):
    if master_name is None:
        master_name = get_master_machine()
    etc_host_file = '/etc/hosts'
    with open(etc_host_file, 'r') as f:
        name_ip_pairs = f.readlines()
    name2ip = {}
    for name_ip_pair in name_ip_pairs:
        pair_list = name_ip_pair.split(' ')
        key = pair_list[1].strip()
        value = pair_list[0]
        name2ip[key] = value
    return name2ip[master_name]
