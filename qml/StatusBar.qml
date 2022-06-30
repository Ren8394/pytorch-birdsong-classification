import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
  id:statusBar
  height: 50
  anchors.left: parent.left
  anchors.top: parent.top
  anchors.right: parent.right
  color: "black"

  RowLayout {
    id: buttonRow
    height: parent.height - 2
    anchors.left: parent.left
    anchors.bottom: parent.bottom
    spacing: 2

    Button {
      id: homeButton
      text: "Home"
      font.pixelSize: 24
      font.bold: true
      Layout.alignment: Qt.AlignLeft | Qt.AlignBottom
      Layout.fillHeight: true
      background: Rectangle {
        implicitWidth: 100
        border.color: "gray"
        border.width: 3
      }
      onClicked: {
        homeButton.background.border.width = 3
        trainButton.background.border.width = 0
        testButton.background.border.width = 0
        homeButton.font.bold = true
        trainButton.font.bold = false
        testButton.font.bold = false
        mainLoader.source = "HomePageView.qml"
      }
    }
    Button {
      id: trainButton
      text: "Train"
      font.pixelSize: 24
      Layout.alignment: Qt.AlignLeft | Qt.AlignBottom
      Layout.fillHeight: true
      background: Rectangle {
        implicitWidth: 100
        border.color: "gray"
        border.width: 0
      }
      onClicked: {
        homeButton.background.border.width = 0
        trainButton.background.border.width = 3
        testButton.background.border.width = 0
        homeButton.font.bold = false
        trainButton.font.bold = true
        testButton.font.bold = false
        mainLoader.source = "TrainPageView.qml"
      }
    }
    Button {
      id: testButton
      text: "Test"
      font.pixelSize: 24
      Layout.alignment: Qt.AlignLeft | Qt.AlignBottom
      Layout.fillHeight: true
      background: Rectangle {
        implicitWidth: 100
        border.color: "gray"
        border.width: 0
      }
      onClicked: {
        homeButton.background.border.width = 0
        trainButton.background.border.width = 0
        testButton.background.border.width = 3
        homeButton.font.bold = false
        trainButton.font.bold = false
        testButton.font.bold = true
        mainLoader.source = "TestPageView.qml"
      }
    }
  } 

  Text {
    text: "TFRI Birdsong Classification APP"
    width: parent.width / 3
    height: parent.height
    anchors.right: parent.right

    font.pixelSize: 28
    color: "white"
    verticalAlignment: Text.AlignVCenter
    horizontalAlignment: Text.AlignHCenter
  } 
}
