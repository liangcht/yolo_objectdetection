import logging
import json

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QAction, QApplication, QMenuBar, QListView, \
    QSizePolicy, QPushButton, QFileDialog
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QSettings, QPoint, QSize, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from gui.utils import center_widget, suppress_errors
from mmod.visual import draw_rect
from mmod.experiment import Experiment
from mmod.detector import Detector


# noinspection PyCallByClass,PyTypeChecker,PyArgumentList
class ImageViewer(QWidget):
    closed = pyqtSignal(int, name="Instance closed")

    def __init__(self, instance, exp=None, source=None, label=None, count=1, parent=None):
        super(ImageViewer, self).__init__(parent=parent)
        self.settings = QSettings('Microsoft', 'taxplorer/viewer')
        self.has_parent = bool(parent)
        self._instance = instance  # the ID to recognize this viewer
        self.exp = exp  # type: Experiment
        self.detector = Detector(self.exp, max_workers=1)
        self._source = source
        self._label = label
        self._count = count
        self._index = 0  # Current index
        self._uid = ''  # Cuurent key uid
        self._key = None
        self._truth = None
        self._handles = None  # type: list

        self._iterator = None
        if self.exp:
            assert isinstance(self.exp, Experiment)
            if label is None:
                assert source is None
                self._iterator = self.exp.imdb.__iter__()
            else:
                self._iterator = self.exp.imdb.iter_label_items(label,
                                                                source=source if source != "total" else None)

        self.cb = QApplication.clipboard()

        self.menu_bar = None  # type: QMenuBar
        self.next_action = None  # type: QAction
        self.figure = None  # type: Figure
        self.canvas = None  # type: FigureCanvas
        self.toolbar = None  # type: NavigationToolbar
        self.list_view = None  # type: QListView
        self.list_model = None  # type: QStandardItemModel
        self.btn_save_orig = None  # type: QPushButton
        self.init_ui()

        # noinspection PyUnresolvedReferences
        self.next_action.triggered.emit()  # Emit the first

    def init_ui(self):
        title = '{}/{} images of {} from {}'.format(self._index + 1,
                                                    self._count,
                                                    self._label if self._label is not None else 'unknown',
                                                    self._source)
        if self._source:
            logging.info(title)
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

        detect_action = QAction('Detect', self)
        detect_action.setShortcut('Ctrl+D')
        # noinspection PyUnresolvedReferences
        detect_action.triggered.connect(self.on_detect)

        self.next_action = QAction('Next', self)
        self.next_action.setShortcut('Right')
        # noinspection PyUnresolvedReferences
        self.next_action.triggered.connect(self.on_next)

        cb_action = QAction('CopyToClipboard', self)
        cb_action.setShortcut('Ctrl+C')
        # noinspection PyUnresolvedReferences
        cb_action.triggered.connect(self.on_cb)

        # no menu if embedded
        if self.has_parent:
            self.addAction(detect_action)
            self.addAction(self.next_action)
            self.addAction(cb_action)
        else:
            self.menu_bar = QMenuBar(self)
            tools_menu = self.menu_bar.addMenu('&Tools')
            tools_menu.addAction(detect_action)
            tools_menu.addAction(self.next_action)
            tools_menu.addAction(cb_action)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.list_model = QStandardItemModel()
        # noinspection PyUnresolvedReferences
        self.list_model.itemChanged.connect(lambda item: self.on_label_changed(item))
        self.list_view = QListView(self)
        self.list_view.setModel(self.list_model)
        self.list_view.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        self.btn_save_orig = QPushButton('Save original', self)
        self.btn_save_orig.setToolTip('Save original image to a file')
        # noinspection PyUnresolvedReferences
        self.btn_save_orig.clicked.connect(self.on_click_save_orig)

        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.toolbar)
        left_vbox.addWidget(self.canvas)

        right_vbox = QVBoxLayout()
        right_vbox.addWidget(self.list_view)
        right_vbox.addWidget(self.btn_save_orig)
        right_vbox.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_vbox)
        main_layout.addLayout(right_vbox)
        if self.menu_bar:
            main_layout.setMenuBar(self.menu_bar)
        self.setLayout(main_layout)

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

    def set_exp(self, exp):
        """Set experiment and source
        :type exp: Experiment
        """
        self.exp = exp
        self.detector.exp = exp
        self._source = self.exp.imdb.path
        self._count = len(self.exp.imdb)
        self._iterator = self.exp.imdb.__iter__()
        self._label = None
        # noinspection PyUnresolvedReferences
        self.next_action.triggered.emit()  # Emit the first

    def _legend(self, labels):
        self.list_model.clear()

        for label in labels:
            item = QStandardItem(label)
            item.setCheckState(Qt.Checked)
            item.setCheckable(True)
            self.list_model.appendRow(item)

    def display(self, key, truth=None, detresults=None, thresh=0.2):
        """Display the image and its ground truth
        :param key: the key to index from imdb
        :param truth: the truth string
        :param detresults: detection results
        :param thresh: threshold for detection results
        """
        truth = json.loads(truth) if truth else []
        if not isinstance(truth, list):
            truth = []
        im = self.exp.imdb.image(key)
        im = im[:, :, (2, 1, 0)]  # BGR to RGB
        ax = self.figure.gca()
        ax.clear()  # clear if previous image is drawn
        all_cls = {}
        ax.imshow(im)
        labels = []
        self._handles = []
        for rect in truth:
            label = rect['class']
            labels.append(label)
            h = draw_rect(ax, all_cls, label, rect['rect'])
            self._handles.append(h)
        self._legend(labels)
        for rect in detresults or []:
            score = rect['conf']
            if score <= thresh:
                continue
            draw_rect(ax, all_cls, rect['class'], rect['rect'], score=score)
        self.canvas.draw()

    @pyqtSlot()
    def on_label_changed(self, item):
        """Called when item is checked/unchecked
        :type item: QStandardItem
        """
        idx = item.row()
        for h in self._handles[idx]:
            h.set_visible(item.checkState())
        self.canvas.draw()

    @pyqtSlot()
    def on_click_save_orig(self):
        if self.exp is None:
            logging.error("No experiment set")
            return
        if not self._key:
            logging.error("No image")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save original image",
            self.settings.value("mru_output_image", "") or "",
            "JPG Files (*.jpg)",
        )
        if not file_name:
            return
        self.settings.setValue("mru_output_image", file_name)

        with suppress_errors():
            with open(file_name, "wb") as f:
                f.write(self.exp.imdb.raw_image(self._key))

    @pyqtSlot()
    def on_detect(self):
        if not self._key:
            return
        with suppress_errors():
            rects = self.detector.detect(self._key)
            self.display(self._key, truth=self._truth, detresults=rects, thresh=0.2)

    @pyqtSlot()
    def on_next(self):
        """Next image
        """
        if self.exp is None:
            logging.error("No experiment set")
            return
        if self._label is None:
            with suppress_errors():
                try:
                    self._key = next(self._iterator)
                    self._truth = None
                    self._uid = self.exp.imdb.uid(self._key)
                    self.display(self._key)
                    self._index += 1
                    self.setWindowTitle('{}/{} images from {} {}'.format(
                        self._index,
                        self._count,
                        self._source,
                        self._uid
                    ))
                except StopIteration:
                    prev_index = self._index
                    self._index = 0
                    assert prev_index == self._count, "Not all indices were iterated: {} != count: {}".format(
                        prev_index, self._count
                    )
                    # wrap to the beginning
                    self._iterator = self.exp.imdb.__iter__()
                    # noinspection PyUnresolvedReferences
                    self.next_action.triggered.emit()
            return
        with suppress_errors():
            try:
                self._key, self._truth = next(self._iterator)
                self._uid = self.exp.imdb.uid(self._key)
                self.display(self._key, self._truth)
                self._index += 1
                self.setWindowTitle('{}/{} images of {} from {} {}'.format(
                    self._index,
                    self._count,
                    self._label,
                    self._source,
                    self._uid
                ))
            except StopIteration:
                prev_index = self._index
                self._index = 0
                assert prev_index == self._count, "Not all indices were iterated: {} != count: {}".format(
                    prev_index, self._count
                )
                # wrap to the beginning
                self._iterator = self.exp.imdb.iter_label_items(
                    self._label,
                    source=self._source if self._source != "total" else None
                )
                # noinspection PyUnresolvedReferences
                self.next_action.triggered.emit()

    @pyqtSlot()
    def on_cb(self):
        """Copy current UID to the clipboard
        """
        if not self._uid:
            return
        with suppress_errors():
            self.cb.setText(self._uid, mode=self.cb.Clipboard)
