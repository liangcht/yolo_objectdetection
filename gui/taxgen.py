import sys
import os
import os.path as op
import logging
import pandas as pd
import json
from collections import OrderedDict

try:
    # noinspection PyPep8Naming
    import cPickle as pickle
except ImportError:
    import pickle

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QAction, QApplication, QMenuBar, QLabel, QFileDialog, \
    QTableView, QHeaderView, QComboBox, QGroupBox, QSpacerItem, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QSettings, QPoint, QSize
from PyQt5 import QtCore

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.insert(0, op.join(op.dirname(this_file), ".."))


from gui.utils import center_widget, suppress_errors, MultiDirectoryDialog, get_version
from gui.pandas_model import PandasModel
from mmod.taxonomy import Taxonomy
from mmod.tax_utils import iterate_tsv_imdb, sample_tax
from mmod.utils import open_with_lineidx, init_logging

os.environ['GLOG_minloglevel'] = '2'  # warnings and errors only

if getattr(sys, 'frozen', False):
    sys_exe_path = op.dirname(sys.executable)
    if op.isdir(op.join(sys_exe_path, 'data')):
        application_path = sys_exe_path
    else:
        application_path = getattr(sys, '_MEIPASS', op.abspath('.'))
else:
    application_path = op.dirname(this_file)


__version__ = get_version()


# noinspection PyCallByClass,PyTypeChecker,PyArgumentList
class TaxGen(QWidget):
    closed = pyqtSignal(int, name="Instance closed")

    def __init__(self, mru_exp_path=None, mru_exp_root=None, instance=-1, parent=None):
        super(TaxGen, self).__init__(parent=parent)
        self._instance = instance
        self.settings = QSettings('Microsoft', 'taxgen')
        self.has_parent = bool(parent)
        self.mru_exp_data = mru_exp_path or self.settings.value("mru_exp_data", op.abspath("."))
        self.mru_exp_root = mru_exp_root or self.settings.value("mru_exp_root", op.abspath("."))
        self.mru_trans_path = self.settings.value("mru_trans_path", op.join(application_path, "label_to_noffset"))

        self.tax = None  # type: Taxonomy
        self.db_df = None  # type: pd.DataFrame

        self.menu_bar = None  # type: QMenuBar
        self.params_box = None  # type: QGroupBox
        self.params_spacer = None  # type: QSpacerItem
        self.lbl_max = None  # type: QLabel
        self.combo_max = None  # type: QComboBox

        self.table_sources = None  # type: QTableView
        self.lbl_last_status = None  # type: QLabel
        self.init_ui()

    def init_ui(self):
        title = 'Taxonomy Generator'
        self.setWindowTitle(title)

        pos = self.settings.value("pos", None)
        size = self.settings.value("size", None)
        need_center = False
        if size and (size.width() < 10 or size.height() < 10):
            size = None
            pos = None
        elif pos and (pos.x() < 0 or pos.y() < 0):
            pos = QPoint(0, 0)
        if size is None or pos is None:
            pos = QPoint(0, 0)
            size = QSize(1024, 768)
            need_center = True
        self.move(pos)
        self.resize(size)
        if need_center:
            self.center()

        load_action = QAction('&Load Yaml', self)
        load_action.setShortcut('Ctrl+L')
        load_action.setStatusTip('Load Yaml Taxonomy')
        # noinspection PyUnresolvedReferences
        load_action.triggered.connect(self.on_load_tax)

        export_action = QAction('&Export Flat', self)
        export_action.setShortcut('Ctrl+E')
        export_action.setStatusTip('Export as flat tsv image database')
        # noinspection PyUnresolvedReferences
        export_action.triggered.connect(self.on_export_flat)

        save_action = QAction('&Save...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save the taxonomy project')
        # noinspection PyUnresolvedReferences
        save_action.triggered.connect(self.on_file_save)

        trans_path_action = QAction('Change translations path', self)
        trans_path_action.setStatusTip('Set global translation yamls path')
        # noinspection PyUnresolvedReferences
        trans_path_action.triggered.connect(self.on_set_trans_path)

        add_source_action = QAction('Add', self)
        add_source_action.setShortcut('Ctrl++')
        add_source_action.setStatusTip('Add more source directories')
        # noinspection PyUnresolvedReferences
        add_source_action.triggered.connect(self.on_add_sources)

        remove_source_action = QAction('Remove Selected', self)
        remove_source_action.setShortcut('Del')
        remove_source_action.setStatusTip('Remove selected sources')
        # noinspection PyUnresolvedReferences
        remove_source_action.triggered.connect(self.on_remove_sources)

        self.menu_bar = QMenuBar(self)
        file_menu = self.menu_bar.addMenu('&File')
        file_menu.addAction(export_action)
        file_menu.addAction(load_action)
        file_menu.addAction(save_action)
        file_menu.addAction(trans_path_action)

        sources_menu = self.menu_bar.addMenu('&Sources')
        sources_menu.addAction(add_source_action)
        sources_menu.addAction(remove_source_action)

        self.params_box = QGroupBox("Parameters")
        params_layout = QHBoxLayout()
        self.lbl_max = QLabel("Max: ")
        params_layout.addWidget(self.lbl_max)
        self.combo_max = QComboBox(self)
        self.combo_max.setEditable(True)
        self.combo_max.lineEdit().setMaxLength(5)
        for i in range(100, 500, 100):
            self.combo_max.addItem(str(i))
        params_layout.addWidget(self.combo_max)

        self.params_spacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        params_layout.addItem(self.params_spacer)
        self.params_box.setLayout(params_layout)

        self.table_sources = QTableView(self)
        self.table_sources.setSortingEnabled(True)

        self.lbl_last_status = QLabel("No taxonomy loaded")
        self.lbl_last_status.setWordWrap(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.params_box)
        main_layout.addWidget(self.table_sources)
        main_layout.addWidget(self.lbl_last_status)
        main_layout.setMenuBar(self.menu_bar)
        self.setLayout(main_layout)

        self.show()

    def closeEvent(self, e):
        """About to close
        :param e: event
        """
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()

        self.closed.emit(self._instance)

    def center(self):
        """Center on screen
        """
        center_widget(self)

    def reload(self):
        """Reload the sources with the taxonomy
        """
        if self.db_df is None or self.tax is None:
            return

        # TODO: add statistics

    @pyqtSlot()
    def on_load_tax(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Taxonomy Yaml",
            self.settings.value("mru_yaml_file", ""),
            "Yaml Files (*.yaml)",
        )

        if not file_name:
            return
        self.settings.setValue("mru_yaml_file", file_name)
        with suppress_errors():
            self.tax = Taxonomy(file_name, trans_path=self.mru_trans_path)
            self.reload()
            self.lbl_last_status.setText(str(self.tax))

    @pyqtSlot()
    def on_set_trans_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Directory | QFileDialog.DirectoryOnly
        dir_name = QFileDialog.getExistingDirectory(
            self, "Select translation directory",
            self.mru_trans_path,
            options=options
        )
        if not dir_name:
            return
        self.mru_trans_path = dir_name
        self.settings.setValue("mru_trans_path", self.mru_trans_path)

        if self.tax is not None:
            self.tax.translation_path = self.mru_trans_path

    @pyqtSlot()
    def on_export_flat(self):
        with suppress_errors():
            assert self.tax is not None, "No taxonomy yaml is loaded"
            assert self.db_df is not None, "No data sources are selected"
        if self.db_df is None or self.tax is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Flat Tsv",
            self.settings.value("mru_tsv_file", "") or self.mru_exp_root or "",
            "TSV Files (*.tsv)",
        )
        if not file_name:
            return

        self.settings.setValue("mru_tsv_file", file_name)
        logging.info("Exporting flat tsv: {}".format(file_name))
        with suppress_errors():
            max_label = int(self.combo_max.currentText())
            inverted_filename = op.splitext(file_name)[0] + ".inverted.label.tsv"
            line_no = 0
            label_lines = OrderedDict()
            with open_with_lineidx(file_name) as fp, open_with_lineidx(inverted_filename) as fpinv:
                for uid, rects, b64 in sample_tax(self.tax, self.db_df["Source"], max_label):
                    if not rects:
                        # we should not get empty rects when sampling a label
                        logging.error("Inconsistent sampling for uid: {} tax: {}".format(uid, self.tax))
                        continue
                    fp.write("{}\t{}\t{}\n".format(
                        uid,
                        json.dumps(rects, separators=(',', ':'), sort_keys=True),
                        b64
                    ))
                    for rect in rects:
                        label = rect['class']
                        if label not in label_lines:
                            label_lines[label] = [line_no]
                        elif label_lines[label][-1] != line_no:
                            label_lines[label].append(line_no)
                    line_no += 1
                for label, lines in label_lines.iteritems():
                    if not lines:
                        continue
                    fpinv.write("{}\t{}\n".format(label, " ".join(map(str, lines))))

    @pyqtSlot()
    def on_add_sources(self):
        sources = MultiDirectoryDialog.get_existing_directories(
            self,
            "Select multiple data source(s)",
            self.settings.value("mru_exp_data", "") or self.mru_exp_data or ""
        )
        if not sources:
            return
        self.mru_exp_data = op.dirname(sources[0])
        self.settings.setValue("mru_exp_data", self.mru_exp_data)
        new_sources = OrderedDict({"Source": [], "Path": []})
        for source in sources:
            with suppress_errors("Error appending {}".format(source)):
                dbs = list(iterate_tsv_imdb(source))
                assert dbs, "No tsv db files found in {}".format(source)
                new_sources["Source"] += dbs
                source = op.normpath(source).replace("\\", "/")
                new_sources["Path"] += [source for _ in dbs]
        with suppress_errors():
            if self.db_df is None:
                self.db_df = pd.DataFrame(new_sources)
            else:
                assert isinstance(self.db_df, pd.DataFrame)
                self.db_df.drop(self.db_df[self.db_df["Path"].isin(new_sources["Path"])].index, inplace=True)
                self.db_df = self.db_df.append(pd.DataFrame(new_sources), ignore_index=True)
            model = PandasModel(self.db_df, checkable=True)
            self.table_sources.setModel(model)
            if not self.db_df.empty:
                self.table_sources.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.reload()

    @pyqtSlot()
    def on_remove_sources(self):
        if self.db_df is None:
            return
        with suppress_errors():
            model = self.table_sources.model()  # type: PandasModel
            checked = []
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                if index.data(role=QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked:
                    checked.append(row)
            if checked:
                self.db_df = self.db_df.drop(self.db_df.index[checked]).reset_index(drop=True)
                model = PandasModel(self.db_df, checkable=True)
                self.table_sources.setModel(model)
                if not self.db_df.empty:
                    self.table_sources.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
                self.reload()

    @pyqtSlot()
    def on_file_save(self):
        with suppress_errors():
            assert self.tax and self.db_df is not None, "Taxonomy project is empty"
        if not self.tax or self.db_df is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save taxonomy project",
            self.settings.value("mru_tax_path", "") or self.mru_exp_root or "",
            "Pickle Files (*.pkl)",
        )
        if not file_name:
            return

        self.settings.setValue("mru_tax_path", file_name)
        with suppress_errors():
            if self.db_df is not None:
                self.db_df.to_csv(op.splitext(file_name)[0] + ".csv")
            with open(file_name, "wb") as fp:
                pickle.dump({
                    "version": __version__,
                    "tax": self.tax,
                    "db_df": self.db_df
                }, fp, protocol=2)


if __name__ == '__main__':
    init_logging()
    app = QApplication(sys.argv)
    ex = TaxGen()
    sys.exit(app.exec_())
