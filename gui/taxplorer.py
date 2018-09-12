from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()

import logging
import sys
import os
import os.path as op

try:
    # noinspection PyPep8Naming
    import cPickle as pickle
except ImportError:
    import pickle

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QLabel, QSpacerItem, \
    QTextEdit, QGroupBox, QVBoxLayout, QGridLayout, QSizePolicy, QTableView, QAbstractItemView, QMenuBar, QAction
from PyQt5.QtCore import pyqtSlot, QSettings, QPoint, QSize


try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.insert(0, op.join(op.dirname(this_file), ".."))


from mmod.imdb import ImageDatabase
from mmod.experiment import Experiment
from mmod.visual import exp_stat
from mmod.utils import init_logging
from gui.pandas_model import PandasModel
from gui.image_viewer import ImageViewer
from gui.taxgen import TaxGen
from gui.utils import center_widget, suppress_errors, get_version

if getattr(sys, 'frozen', False):
    sys_exe_path = op.dirname(sys.executable)
    if op.isdir(op.join(sys_exe_path, 'data')):
        application_path = sys_exe_path
    else:
        application_path = getattr(sys, '_MEIPASS', op.abspath('.'))
else:
    application_path = op.dirname(this_file)
os.chdir(application_path)

__version__ = get_version()

os.environ['GLOG_minloglevel'] = '2'  # warnings and errors only


# noinspection PyCallByClass,PyTypeChecker,PyArgumentList
class App(QWidget):

    def __init__(self):
        super(App, self).__init__()
        self.title = 'Taxonomy Explorer'
        self.settings = QSettings('Microsoft', 'taxplorer')
        self.imdb = None  # type: ImageDatabase
        self.exp = None  # type: Experiment
        self.caffenet = self.settings.value("mru_caffenet", op.abspath("./data/test.prototxt"))
        self.caffemodel = self.settings.value("mru_caffemodel",
                                              op.abspath("./data/snapshot/model_iter_32264.caffemodel"))
        self.mru_exp_data = self.settings.value("mru_exp_data", op.abspath("."))
        self.mru_imdb_path = self.settings.value("mru_imdb_path", None)
        self.mru_exp_root = self.settings.value("mru_exp_root", op.expanduser("~/Desktop"))
        self.exp_df = None  # type: pandas.DataFrame
        self.viewers = {}
        self.gens = set()

        self.menu_bar = None

        self.db_box = None
        self.exp_box = None

        self.btn_dir = None
        self.btn_file = None
        self.edt_imdb_path = None
        self.db_spacer = None
        self.btn_exp_data = None
        self.lbl_exp_data = None
        self.btn_caffenet = None
        self.edt_caffenet = None
        self.btn_caffemodel = None
        self.edt_caffemodel = None
        self.btn_reload = None
        self.btn_reset = None
        self.exp_spacer = None

        self.lbl_last_status = None
        self.table_imdb = None
        self.viewer = None
        self.init_ui()
        logging.info("version: {} path: {}".format(__version__, application_path))

    def init_ui(self):
        self.setWindowTitle(self.title)

        pos = self.settings.value("pos", None)
        size = self.settings.value("size", None)
        if size and (size.width() < 10 or size.height() < 10):
            size = None
            pos = None
        elif pos and (pos.x() < 0 or pos.y() < 0):
            pos = QPoint(0, 0)
        need_center = False
        if size is None or pos is None:
            pos = QPoint(0, 0)
            size = QSize(1024, 768)
            need_center = True
        self.move(pos)
        self.resize(size)
        if need_center:
            self.center()

        save_action = QAction('&Save...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save the experiment and experiment metrics')
        # noinspection PyUnresolvedReferences
        save_action.triggered.connect(self.on_file_save)

        open_action = QAction('&Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open a previous experiment')
        # noinspection PyUnresolvedReferences
        open_action.triggered.connect(self.on_file_open)

        new_action = QAction('&New...', self)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('Create a new taxonomy')
        # noinspection PyUnresolvedReferences
        new_action.triggered.connect(self.on_file_new)

        self.menu_bar = QMenuBar(self)
        file_menu = self.menu_bar.addMenu('&File')
        file_menu.addAction(save_action)
        file_menu.addAction(open_action)
        file_menu.addAction(new_action)

        self.db_box = QGroupBox("DB Setup")
        db_layout = QVBoxLayout()

        self.btn_dir = QPushButton('Directory as image database', self)
        db_layout.addWidget(self.btn_dir)
        self.btn_dir.setToolTip('Choose a directory with images')
        # noinspection PyUnresolvedReferences
        self.btn_dir.clicked.connect(self.on_click_dir_imdb)

        self.btn_file = QPushButton('File as image database', self)
        db_layout.addWidget(self.btn_file)
        self.btn_file.setToolTip('Choose a tsv/prototxt/image file')
        # noinspection PyUnresolvedReferences
        self.btn_file.clicked.connect(self.on_click_file_imdb)

        self.edt_imdb_path = QTextEdit(self.mru_imdb_path if self.mru_imdb_path else "")
        self.edt_imdb_path.setReadOnly(True)
        self.edt_imdb_path.setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: none; }")
        db_layout.addWidget(self.edt_imdb_path)

        self.db_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        db_layout.addItem(self.db_spacer)
        self.db_box.setLayout(db_layout)

        self.exp_box = QGroupBox("Experiment")
        exp_layout = QVBoxLayout()

        self.btn_exp_data = QPushButton('Experiment data directory', self)
        exp_layout.addWidget(self.btn_exp_data)
        self.btn_exp_data.setToolTip('Choose the data directory')
        # noinspection PyUnresolvedReferences
        self.btn_exp_data.clicked.connect(self.on_click_dir_exp)

        self.lbl_exp_data = QLabel(self.mru_exp_data if self.mru_exp_data else "")
        exp_layout.addWidget(self.lbl_exp_data)
        self.lbl_exp_data.setWordWrap(True)

        self.btn_caffenet = QPushButton('Caffenet', self)
        exp_layout.addWidget(self.btn_caffenet)
        self.btn_caffenet.setToolTip('Choose the test caffenet for the experiment')
        # noinspection PyUnresolvedReferences
        self.btn_caffenet.clicked.connect(self.on_click_caffenet)

        self.edt_caffenet = QTextEdit(self.caffenet if self.caffenet else "")
        exp_layout.addWidget(self.edt_caffenet)
        self.edt_caffenet.setReadOnly(True)
        self.edt_caffenet.setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: none; }")

        self.btn_caffemodel = QPushButton('Caffemodel', self)
        exp_layout.addWidget(self.btn_caffemodel)
        self.btn_caffemodel.setToolTip('Choose the caffemodel for the experiment')
        # noinspection PyUnresolvedReferences
        self.btn_caffemodel.clicked.connect(self.on_click_caffemodel)

        self.edt_caffemodel = QTextEdit(self.caffemodel if self.caffemodel else "")
        exp_layout.addWidget(self.edt_caffemodel)
        self.edt_caffemodel.setReadOnly(True)
        self.edt_caffemodel.setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: none; }")

        self.btn_reload = QPushButton('Reload', self)
        exp_layout.addWidget(self.btn_reload)
        self.btn_reload.setToolTip('Relaod imdb and experiment')
        # noinspection PyUnresolvedReferences
        self.btn_reload.clicked.connect(self.on_click_reload)

        self.btn_reset = QPushButton('Reset', self)
        exp_layout.addWidget(self.btn_reset)
        self.btn_reset.setToolTip('Reset experiment to empty')
        # noinspection PyUnresolvedReferences
        self.btn_reset.clicked.connect(self.on_click_reset)

        self.exp_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        exp_layout.addItem(self.exp_spacer)
        self.exp_box.setLayout(exp_layout)

        self.lbl_last_status = QLabel("No db loaded")
        self.lbl_last_status.setWordWrap(True)

        self.table_imdb = QTableView(self)
        self.table_imdb.hide()
        self.table_imdb.setSortingEnabled(True)
        self.table_imdb.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table_imdb.setSelectionMode(QAbstractItemView.SingleSelection)
        # noinspection PyUnresolvedReferences
        self.table_imdb.doubleClicked.connect(self.on_double_click_table)

        # the main viewer for dbs with no truth
        self.viewer = ImageViewer(-1, parent=self)
        self.viewer.hide()

        main_layout = QGridLayout()
        main_layout.setMenuBar(self.menu_bar)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 4)
        main_layout.setRowStretch(0, 4)
        main_layout.setRowStretch(1, 4)
        main_layout.setRowStretch(2, 1)

        main_layout.addWidget(self.db_box, 0, 0)
        main_layout.addWidget(self.exp_box, 1, 0)
        main_layout.addWidget(self.lbl_last_status, 2, 0, 1, 2)
        main_layout.addWidget(self.table_imdb, 0, 1, 2, 1)
        main_layout.addWidget(self.viewer, 0, 1, 2, 1)

        self.setLayout(main_layout)

        self.setAcceptDrops(True)
        self.show()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        found_imdb = False
        found_caffemodel = False
        found_caffenet = False
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            is_dir = op.isdir(path)
            is_file = op.isfile(path)
            if not path or not is_file and not is_dir:
                if path:
                    logging.debug("ignore non-existent file: {}".format(path))
                continue

            if is_file:
                ext = op.splitext(path)[1]
                if ext == ".caffemodel":
                    if found_caffemodel:
                        logging.info("Ignore {} because {} is already used".format(path, self.caffemodel))
                    else:
                        self.set_caffemodel(path)
                elif ext == ".prototxt":
                    # this can be an imdb or a caffenet
                    if found_imdb:
                        if not found_caffenet:
                            found_caffenet = True
                            self.set_caffenet(path)
                            continue
                    if found_caffenet:
                        logging.info("Ignore {} because {} and {} is already found".format(
                            path, self.caffenet, self.imdb.path
                        ))
                        continue

                    # first try as db
                    try:
                        # see if there are images here
                        self.set_imdb(path)
                        found_imdb = True
                    except Exception as e:
                        if not found_caffenet:
                            self.set_caffenet(path)
                            found_caffenet = True
                        else:
                            logging.error("{} was not a valid imdb. error: {}".format(path, e))
                    continue

            if found_imdb:
                logging.info("Ignore {} because {} and {} is already found".format(
                    path, self.caffenet, self.imdb.path
                ))
                continue

            try:
                # see if there are images here
                self.set_imdb(path)
                found_imdb = True
            except Exception as e:
                logging.error("{} was not a valid imdb. error: {}".format(path, e))

            if found_imdb and (self.imdb.is_image or self.imdb.is_directory or self.imdb.is_tsv):
                with suppress_errors():
                    self.load_exp(reset=True)

    def set_imdb(self, path):
        self.chdir()

        self.imdb = ImageDatabase(path)
        self.mru_imdb_path = path
        self.lbl_last_status.setText(str(self.imdb))
        self.edt_imdb_path.setText(self.mru_imdb_path)

    def set_caffenet(self, path):
        self.caffenet = path
        if self.caffenet:
            self.settings.setValue("mru_caffenet", self.caffenet)
        self.edt_caffenet.setText(self.caffenet)

    def set_caffemodel(self, path):
        self.caffemodel = path
        if self.caffemodel:
            self.settings.setValue("mru_caffemodel", self.caffemodel)
        self.edt_caffemodel.setText(self.caffemodel)

    def closeEvent(self, e):
        """About to close
        :param e: event
        """
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()

    def center(self):
        """Center on screen
        """
        center_widget(self)

    @pyqtSlot()
    def on_double_click_table(self):
        row = col = None
        for index in self.table_imdb.selectedIndexes():
            row, col = index.row(), index.column()
            break

        with suppress_errors():
            self.visualize_label(row, col)

    @pyqtSlot()
    def on_click_file_imdb(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Single File",
            self.settings.value("mru_imdb_file", ""),
            ("All Files (*);;"
             "TSV Files (*.tsv);;"
             "Caffe Training(*.prototxt)"),
            options=options
        )
        if not file_name:
            return

        self.mru_imdb_path = file_name
        self.settings.setValue("mru_imdb_file", self.mru_imdb_path)
        self.settings.setValue("mru_imdb_path", self.mru_imdb_path)
        with suppress_errors():
            self.load_imdb()
            if self.imdb.is_prototxt or self.imdb.is_tsv:
                self.load_exp(reset=True)

    @pyqtSlot()
    def on_click_dir_imdb(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Directory | QFileDialog.DirectoryOnly
        dir_name = QFileDialog.getExistingDirectory(
            self, "Select image data directory",
            self.settings.value("mru_imdb_dir", ""),
            options=options
        )
        if not dir_name:
            return

        self.mru_imdb_path = dir_name
        self.settings.setValue("mru_imdb_dir", self.mru_imdb_path)
        self.settings.setValue("mru_imdb_path", self.mru_imdb_path)
        with suppress_errors():
            self.load_imdb()

    @pyqtSlot()
    def on_click_dir_exp(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Directory | QFileDialog.DirectoryOnly
        dir_name = QFileDialog.getExistingDirectory(
            self, "Select data directory",
            self.settings.value("mru_exp_data", ""),
            options=options
        )
        if not dir_name:
            return

        self.mru_exp_data = dir_name
        self.settings.setValue("mru_exp_data", self.mru_exp_data)
        self.lbl_exp_data.setText(self.mru_exp_data)
        if self.exp:
            with suppress_errors():
                self.exp.data = self.mru_exp_data

    @pyqtSlot()
    def on_click_caffenet(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Caffenet File",
            self.settings.value("mru_caffenet", ""),
            "Caffenet (*.prototxt)",
        )
        if not file_name:
            return

        self.set_caffenet(file_name)

    @pyqtSlot()
    def on_click_caffemodel(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Caffemodel File",
            self.settings.value("mru_caffemodel", ""),
            "Caffemodel (*.caffemodel)",
        )
        if not file_name:
            return

        self.set_caffemodel(file_name)

    @pyqtSlot()
    def on_click_reload(self):
        """Reload imdb and exp
        """
        with suppress_errors():
            self.load_imdb()
            self.load_exp()

    @pyqtSlot()
    def on_click_reset(self):
        self.set_caffemodel('')
        self.set_caffenet('')
        if self.imdb is None:
            return
        with suppress_errors():
            self.load_exp()

    @pyqtSlot()
    def on_file_open(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Project Exported pkl File",
            self.settings.value("mru_pkl_file", self.settings.value("mru_exp_root", "")),
            "pkl File (*.pkl)",
        )
        if not file_name:
            return

        self.settings.setValue("mru_pkl_file", op.dirname(file_name))
        with suppress_errors("Loading saved project"):
            self.load(file_name)

    @pyqtSlot()
    def on_file_new(self):
        gen = TaxGen(self.mru_exp_data, self.mru_exp_root)
        self.gens.add(gen)
        gen.show()

    @pyqtSlot()
    def on_file_save(self):
        with suppress_errors():
            assert self.exp, "No experiment is loaded to export"
        if not self.exp:
            return

        options = QFileDialog.Options()
        options |= QFileDialog.Directory | QFileDialog.DirectoryOnly
        dir_name = QFileDialog.getExistingDirectory(
            self, "Select experiment output root directory",
            self.settings.value("mru_exp_root", ""),
            options=options
        )
        if not dir_name:
            return

        self.mru_exp_root = dir_name
        self.settings.setValue("mru_exp_root", self.mru_exp_root)
        with suppress_errors():
            self.exp.root = self.mru_exp_root
            if self.exp_df is not None:
                self.exp_df.to_csv(self.exp.path + ".metrics.csv")
            with open(self.exp.path + ".pkl", "wb") as fp:
                pickle.dump({
                    "version": __version__,
                    "exp": self.exp,
                    "exp_df": self.exp_df
                }, fp, protocol=2)

    @pyqtSlot()
    def on_viewer_closed(self, instance):
        """Signal to clean up a closed viewer
        :type instance: int
        """
        with suppress_errors():
            assert instance < len(self.viewers), "Invalid viewer instance: {} >= {}".format(
                instance, len(self.viewers)
            )
            self.viewers[instance] = None

    def chdir(self):
        """Change directory to data root
        """
        if not self.mru_exp_data:
            logging.error("data directory not set for prototxt db")
        else:
            with suppress_errors():
                # Prototxt may depend on data path
                data_dir = op.join(self.mru_exp_data, "data")
                assert op.isdir(data_dir), "No 'data' directory found in {}".format(self.mru_exp_data)
                os.chdir(self.mru_exp_data)

    def load_imdb(self):
        """Laod imdb
        """
        assert self.mru_imdb_path, "No IMDB is given"
        assert op.isdir(self.mru_imdb_path) or op.isfile(self.mru_imdb_path), \
            "Invalid IMDB at {}".format(self.mru_imdb_path)

        self.set_imdb(self.mru_imdb_path)

    def load_exp(self, reset=False):
        """Laod experiment
        :param reset: if should automatically re-set experiment's caffenet and caffemodel based on imdb
        """
        assert self.imdb, "Image database not given"
        self.chdir()
        if reset:
            self.set_caffemodel('')
            self.set_caffenet('')
        self.exp = Experiment(self.imdb, self.caffenet, self.caffemodel,
                              root=self.mru_exp_root, data=self.mru_exp_data, reset=reset)
        if reset:
            # noinspection PyBroadException
            try:
                # populate with potentially changed test
                self.set_caffemodel(self.exp.caffemodel)
                self.set_caffenet(self.exp.caffenet)
            except Exception:
                logging.info("Caffenet/caffemodel not set")

        self.lbl_last_status.setText(str(self.exp))
        if not self.mru_exp_root:
            self.mru_exp_root = self.exp.root
            self.settings.setValue("mru_exp_root", self.mru_exp_root)

        self.exp_df = None
        if self.imdb.is_directory or self.imdb.is_image:
            self.table_imdb.hide()
            self.viewer.set_exp(self.exp)
            self.viewer.show()
            return

        self.viewer.hide()
        self.exp_df = exp_stat(self.exp).sort_index()
        model = PandasModel(self.exp_df)
        self.table_imdb.setModel(model)
        self.table_imdb.show()

    def visualize_db(self):
        instance = len(self.viewers)
        viewer = ImageViewer(instance, self.exp, self.imdb.name)
        viewer.closed.connect(lambda: self.on_viewer_closed(instance))
        viewer.show()
        self.viewers[instance] = viewer

    def visualize_label(self, row, col):
        assert row is not None and col is not None, "No label selected"

        source = self.exp_df.columns[col]
        label = self.exp_df.index[row]
        count = self.exp_df[source][label]
        assert count > 0, "{} has no images to show in {}".format(label, source)
        instance = len(self.viewers)
        viewer = ImageViewer(instance, self.exp, source, label, count)
        viewer.closed.connect(lambda: self.on_viewer_closed(instance))
        viewer.show()
        self.viewers[instance] = viewer

    def load(self, path):
        """Load project from pickle
        """
        with open(path, 'rb') as fp:
            me = pickle.load(fp)
        self.exp = me["exp"]  # type: Experiment
        with suppress_errors("output root directory may have changed"):
            self.mru_exp_root = self.exp.root
            self.chdir()
        self.imdb = self.exp.imdb
        self.exp_df = me["exp_df"]
        with suppress_errors("caffenet may no longer exist"):
            self.caffenet = self.exp.caffenet
        with suppress_errors("caffemodel may no longer exist"):
            self.caffemodel = self.exp.caffemodel
        with suppress_errors("data directory may have changed"):
            self.mru_exp_data = self.exp.data
        self.lbl_exp_data.setText(self.mru_exp_data)
        self.edt_caffemodel.setText(self.caffemodel)
        self.edt_caffenet.setText(self.caffenet)
        self.lbl_exp_data.setText(self.mru_exp_data)
        if self.exp_df is not None:
            model = PandasModel(self.exp_df)
            self.table_imdb.setModel(model)
            self.table_imdb.show()
        # Update the status label
        self.lbl_last_status.setText(str(self.exp))


if __name__ == '__main__':
    init_logging()
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
