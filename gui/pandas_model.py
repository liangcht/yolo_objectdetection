from PyQt5 import QtCore

import pandas as pd


# noinspection PyMethodOverriding
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), checkable=False, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df
        self._is_checked = None
        if checkable:
            self._is_checked = {
                idx: False for idx in self._df.index
            }

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def flags(self, index):
        if not index.isValid():
            return
        if not self._is_checked or index.column() != 0:
            return QtCore.QAbstractTableModel.flags(self, index)
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return QtCore.QVariant()

        if role == QtCore.Qt.CheckStateRole and index.column() == 0 and self._is_checked:
            if self._is_checked[self._df.index[index.row()]]:
                return QtCore.Qt.Checked
            return QtCore.Qt.Unchecked
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        if role == QtCore.Qt.CheckStateRole and index.column() == 0 and self._is_checked:
            self._is_checked[row] = value == QtCore.Qt.Checked
            return True
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        # noinspection PyUnresolvedReferences
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending=order == QtCore.Qt.AscendingOrder, inplace=True)
        # noinspection PyUnresolvedReferences
        self.layoutChanged.emit()
