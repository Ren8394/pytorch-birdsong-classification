import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.qmlmodels

import DFModel

RowLayout {
  id: testPageMainLayout
  anchors.fill: parent
  spacing: 10

  ColumnLayout {
    id: leftCol
    Layout.fillHeight: true
    Layout.topMargin: 20
    Layout.bottomMargin: 20

    Text {
      Layout.fillWidth: true
      Layout.bottomMargin: 30
      text: qsTr("AutoEncoder Classifier")
      font.pixelSize: 32
      color: "yellow"
      horizontalAlignment: Text.AlignHCenter
    }

    Text {
      id: weightFileText
      Layout.fillWidth: true
      text: qsTr("AEC_20xxxx_xxxx.pth")
      font.pixelSize: 18
      color: "lightgray"
      horizontalAlignment: Text.AlignHCenter
    }

    Text {
      id: fileText
      Layout.fillWidth: true
      text: qsTr("WWWWWW_XXXXXXXX_XXXXXX.wav")
      font.pixelSize: 18
      color: "lightgray"
      horizontalAlignment: Text.AlignHCenter
    }

    Rectangle {
      width: 240
      height: 240
      border.color: "white"
      border.width: 2
      Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

      Image {
        id: specImage
        width: 240 - 2 * parent.border.width
        height: 240 - 2 * parent.border.width
        anchors.centerIn: parent
        source: "../assets/spec.png"
        fillMode: Image.PreserveAspectFit
      }
    }

    Row {
      id: mediaRow
      width: 240
      Layout.alignment: Qt.AlignHCenter
      Layout.topMargin: 10
      spacing: 10

      Button {
        id: playButton
        anchors.leftMargin: 80
        width: 50
        background: Rectangle {
          width: 50
          height: 50
          color: "lightgreen"
          radius: 25

          Text {
            text: "Play"
            font.pixelSize: 16
            anchors.centerIn: parent
          }
        }
        onClicked: {
          console.log("play")
        }
      }

      Button {
        id: pauseButton
        anchors.rightMargin: 80
        width: 50
        background: Rectangle {
          width: 50
          height: 50
          color: "lightgreen"
          radius: 25

          Text {
            text: "Pause"
            font.pixelSize: 16
            anchors.centerIn: parent
          }
        }
        onClicked: {
          console.log("pause")
        }
      }
    }
  }

  TableView {
    id: resultTableView
    width: parent.width / 3
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter
    Layout.topMargin: 20
    Layout.bottomMargin: 20
    columnSpacing: 0
    rowSpacing: 0
    boundsBehavior: Flickable.StopAtBounds

    model: DFModel{}

    delegate: DelegateChooser {
      DelegateChoice {
        column: 0
        delegate: Rectangle {
          implicitWidth: resultTableView.width / 3
          implicitHeight: 50
          border.width: 1

          Text {
            text: qsTr(model.display)
            anchors.centerIn: parent
            font.pixelSize: 18
          }
        }
      }
      DelegateChoice {
        column: 1
        delegate: Rectangle {
          implicitWidth: resultTableView.width / 3
          implicitHeight: 50
          border.width: 1
          color: model.display >= 0.5 ? "lightgreen": "pink"

          Text {
            text: model.display
            anchors.centerIn: parent
            font.pixelSize: 18
          }
        }
      }
      DelegateChoice {
        column: 2
        delegate: CheckBox {
          checked: false
          onToggled: model.display = checked

          indicator: Rectangle {
            implicitWidth: 18
            implicitHeight: 18
            anchors.centerIn: parent
            radius: 3

            Rectangle {
              width: 12
              height: 12
              anchors.centerIn: parent
              radius: 2
              color: checked ? "red": "black"
              visible: checked
            }
          }
        }
      }
      DelegateChoice {
        column: 3
        delegate: TextField {
          id: textField
          verticalAlignment: TextInput.AlignVCenter
          selectByMouse: true
          implicitWidth: resultTableView.width / 3 - 20
          color: "black"
          font.pixelSize: 16
          background: Rectangle {
            border.width: textField.activeFocus ? 2: 1
            color: "white"
            border.color: textField.activeFocus ? "blue": "black"
          }
          onAccepted: model.display = text
        }
      }
    }
  }

  ColumnLayout {
    width: parent.width / 6
    Layout.fillHeight: true
    Layout.leftMargin: 30
    Layout.topMargin: 20
    Layout.bottomMargin: 20
    Layout.rightMargin: 20
    spacing: 20

    Row {
      height: 20
      spacing: 20
      Text {
        text: qsTr("Load Model Weight")
        width: 150
        font.pixelSize: 18
        color: "white"
        verticalAlignment: Text.AlignVCenter
      }
      Button {
        id: weightButton
        text: "Load Model"
        font.pixelSize: 16
        background: Rectangle {
          implicitWidth: 120
          Layout.fillHeight: true
          color: "lightgray"
          radius: 5
        }
        onClicked: {
          console.log("load model")
        }
      }
    }

    Row {
      height: 20
      spacing: 20
      Text {
        text: qsTr("Load Audio")
        width: 150
        font.pixelSize: 18
        color: "white"
        verticalAlignment: Text.AlignVCenter
      }
      Button {
        id: selectButton
        text: "Load Audio"
        font.pixelSize: 16
        background: Rectangle {
          implicitWidth: 120
          Layout.fillHeight: true
          color: "lightgray"
          radius: 5
        }
        onClicked: {
          console.log("load audio")
        }
      }
    }

    Rectangle {
      Layout.fillWidth: true
      Layout.fillHeight: true
      color: Qt.rgba(0, 0, 0, 0)
    }

    Row {
      Layout.fillWidth: true
      Layout.rightMargin: 10
      Layout.bottomMargin: 10
      height: 20
      layoutDirection: Qt.RightToLeft
      spacing: 20

      Button {
        id: doneButton
        text: "Done"
        font.pixelSize: 16
        background: Rectangle {
          implicitWidth: 100
          Layout.fillHeight: true
          color: "lightgray"
          radius: 5
        }
        onClicked: {
          console.log("done")
        }
      }

      Button {
        id: startButton
        text: "Start"
        font.pixelSize: 16
        background: Rectangle {
          implicitWidth: 100
          Layout.fillHeight: true
          color: "lightgray"
          radius: 5
        }
        onClicked: {
          console.log("start")
        }
      }

      Text {
        height: 20
        text: qsTr("Auto")
        font.pixelSize: 18
        color: "white"
        verticalAlignment: Text.AlignVCenter
      }

      CheckBox {
        id: autoCheckBox
        Layout.alignment: Qt.AlignVCenter
        width: 20
        height: 20
      }
    }
  }
}
