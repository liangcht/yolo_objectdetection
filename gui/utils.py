from contextlib import contextmanager
import traceback
import logging

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtWidgets import QFileDialog, QTreeView, QAbstractItemView, QListView, QFileSystemModel


def center_widget(widget):
    """Place the widget in the center of current screen
    :param widget:
    """
    frame_gm = widget.frameGeometry()
    # noinspection PyArgumentList
    screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
    # noinspection PyArgumentList
    center_point = QApplication.desktop().screenGeometry(screen).center()
    frame_gm.moveCenter(center_point)
    widget.move(frame_gm.topLeft())


@contextmanager
def suppress_errors(msg=None):
    error_dialog = QMessageBox()
    error_dialog.setStandardButtons(QMessageBox.Ok)
    error_dialog.setIcon(QMessageBox.Critical)
    error_dialog.setEscapeButton(QMessageBox.Ok)

    # noinspection PyBroadException
    try:
        yield
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("{}".format(tb))
        error_dialog.setWindowTitle("Error!")
        error_dialog.setText(str(e) + "" if not msg else " {}".format(msg))
        error_dialog.setDetailedText(tb)
        error_dialog.exec_()
    finally:
        pass


class MultiDirectoryDialog(QFileDialog):
    def __init__(self, *args):
        QFileDialog.__init__(self, *args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)

        for view in self.findChildren((QListView, QTreeView)):
            if isinstance(view.model(), QFileSystemModel):
                view.setSelectionMode(QAbstractItemView.ExtendedSelection)

    @staticmethod
    def get_existing_directories(*args):
        """Get mutiple directories
        :param args: arguments to create the actual QFileDialog
        :rtype: list
        """
        dlg = MultiDirectoryDialog(*args)
        if dlg.exec_():
            return dlg.selectedFiles()
        return []


def get_version():
    return '1.0.0'
