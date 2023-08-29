# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'model_selectionUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from model_registry import ModelRegistry
import os, json

class Ui_MainWindow(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1242, 789)
        self.centralwidget = QtWidgets.QWidget(Dialog)

        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(700, 680, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)

        self.modeltableW = QtWidgets.QTableWidget(self.centralwidget)
        self.modeltableW.setGeometry(QtCore.QRect(20, 140, 551, 561))
        self.modeltableW.selectionModel().selectionChanged.connect(self.update_label)

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(590, 140, 301, 481))
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_1.setContentsMargins(0, 0, 0, 0)

        self.active_learn = QtWidgets.QCheckBox("Active-learning", self.verticalLayoutWidget_2)
        self.active_learn.stateChanged.connect(self.IsActiveLearn)
        self.verticalLayout_1.addWidget(self.active_learn)

        self.lt = QtWidgets.QCheckBox("LT",self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.lt)
        self.tronly = QtWidgets.QCheckBox("TrOnly", self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.tronly)
        self.trchass = QtWidgets.QCheckBox("TrChass", self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.trchass)
        self.trflat = QtWidgets.QCheckBox("TrFlat",self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.trflat)
        self.trtrail = QtWidgets.QCheckBox("TrTrail", self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.trtrail)
        self.trreefer = QtWidgets.QCheckBox("TrReefer", self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.trreefer)
        self.bus = QtWidgets.QCheckBox("Bus",self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.bus)
        self.trcont = QtWidgets.QCheckBox("TrCont", self.verticalLayoutWidget_2)
        self.verticalLayout_1.addWidget(self.trcont)

        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(900, 140, 271, 481))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.trtank = QtWidgets.QCheckBox("TrTank", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.trtank)
        self.const = QtWidgets.QCheckBox("Const", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.const)
        self.waste = QtWidgets.QCheckBox("Waste", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.waste)
        self.o = QtWidgets.QCheckBox("O", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.o)
        self.vp = QtWidgets.QCheckBox("VP", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.vp)
        self.rvc = QtWidgets.QCheckBox("RVC", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.rvc)
        self.sv = QtWidgets.QCheckBox("SV", self.verticalLayoutWidget_3)
        self.verticalLayout_2.addWidget(self.sv)

        self.choosed_model = QtWidgets.QLabel(self.centralwidget)
        self.choosed_model.setGeometry(QtCore.QRect(30, 60, 641, 41))

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 20, 641, 41))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("Sort_by: ",self.horizontalLayoutWidget)
        self.horizontalLayout.addWidget(self.label)
        self.Confi = QtWidgets.QRadioButton("Confi", self.horizontalLayoutWidget)
        self.horizontalLayout.addWidget(self.Confi)
        self.Classcount = QtWidgets.QRadioButton("class count",self.horizontalLayoutWidget)
        self.horizontalLayout.addWidget(self.Classcount)
        self.speed = QtWidgets.QRadioButton("Speed",self.horizontalLayoutWidget)
        self.horizontalLayout.addWidget(self.speed)
        self.speed.clicked.connect(self.SortbySpeed)
        self.Confi.clicked.connect(self.Sortbymap50)
        self.Classcount.clicked.connect(self.SortbyClassCount)
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(self.Result) # type: ignore
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.read_data()
        self.message = None

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ModelSelection"))
        self.choosed_model.setText(_translate("Dialog", "Model choose:"))

    def set_checkboxes(self):
        # 1st: show set all checkboxes as checked.
        self.checkBoxes = []
        self.checkBoxes.extend(
            [
                self.lt,
                self.tronly,
                self.trchass,
                self.trflat,
                self.sv,
                self.trtrail,
                self.trcont,
                self.rvc,
                self.trtank,
                self.const,
                self.waste,
                self.o,
                self.vp,
                self.bus,
                self.trreefer,
            ]
        )
        for checkbox in self.checkBoxes:
            checkbox.setChecked(True)
        self.active_learn.setChecked(True)

    def read_data(self):
        # Show model selection list in the table widgets.
        # load datat to the table.
        parent_path = os.getcwd()
        folder_path = os.path.join(parent_path, "models","detection")    # this is the model directory path
        Load_data = ModelRegistry(folder_path).create_dataframe()
        self.modeldataset = Load_data.values.tolist()
        self.modelHeader = Load_data.columns.values.tolist()
        self.show_Modellist()

    def show_Modellist(self):
        self.modeltableW.setColumnCount(len(self.modeldataset[0]))       # set table col count acording to dataframe.
        self.modeltableW.setRowCount(len(self.modeldataset))
        self.modeltableW.setHorizontalHeaderLabels(self.modelHeader)
        for row_idx, row_data in enumerate(self.modeldataset):
            for col_idc, col_data in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(str(col_data))
                self.modeltableW.setItem(row_idx,col_idc,item)
        self.set_checkboxes()

    def update_label(self):
        row = self.modeltableW.currentRow()
        col_num = self.modelHeader.index("folder")
        self.message = self.modeltableW.item(row, col_num).text()  # Message is the frame count from the csv.
        # update label 
        self.choosed_model.setText("Model choose: %s" %self.message)
        self.update_checkboxes(self.message)

    def update_checkboxes(self,message):
        #open and read json file
        Jason_path = os.path.join(os.getcwd(), "models","detection",message,"metadata.json")    # this is the model directory path
        with open(Jason_path, 'r') as file:
            data = json.load(file)
        # Get the set of classes from the JSON data
        json_classes = set(data['classes'].values())
        for checkbox in self.checkBoxes:
            class_name = checkbox.text().lower()
            checkbox.setChecked(class_name in json_classes)
            checkbox.setVisible(class_name in json_classes)

    def IsActiveLearn(self):
        # Check if the actice_learn is checked or not.
        if not self.active_learn.isChecked():
            for i in self.checkBoxes: 
                i.hide()
                i.setChecked(False)
        else: 
            for i in self.checkBoxes: 
                i.setChecked(True)
                i.show()

    def SortbySpeed(self):
        # sort data by its speed
        try:
            print("Sort by Speed trigered.")
            col_num = self.modelHeader.index("inf speed (ms)")
            self.modeldataset.sort(key=lambda x: float(x[col_num]), reverse=True)
            self.show_Modellist()
        except AttributeError as err:
            print("Can't sort when here is no data!")

    def SortbyClassCount(self):
        # sort data by it class count
        try:
            print("Sort by class count trigered.")
            col_num = self.modelHeader.index("num of classes")
            self.modeldataset.sort(key=lambda x: float(x[col_num]), reverse=True)
            self.show_Modellist()
        except AttributeError as err:
            print("Can't sort when here is no data!")

    def Sortbymap50(self):
        # sort data by it confidence.
        try:
            print("Sort by confi trigered.")
            col_num = self.modelHeader.index("mAP50")
            self.modeldataset.sort(key=lambda x: float(x[col_num]), reverse = True)
            self.show_Modellist()
        except AttributeError as err:
            print("Can't sort when here is no data!")

    def Result(self):
        # return Result_list
        final_Result= {"model": self.message}
        final_Result["active_learn"] = self.active_learn.isChecked()
        for item in self.checkBoxes:
            final_Result[item.text().lower()] = item.isChecked()
        self.mydic = final_Result
        return final_Result


def showModelSel():
    #MainWindow = QtWidgets.QMainWindow()
    Dialog = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(Dialog)
    if Dialog.exec_():
        #ret=ui.Xjson
        pass
    # print(ui.mydic)
    return ui.mydic

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())