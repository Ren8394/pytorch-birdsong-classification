import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Window

ApplicationWindow {
  id: mainWindow
  title: qsTr("TFRI Birdsong Classification APP")
  width: 1280
  height: 720
  visible: true

  // Flags
  flags: Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowTitleHint | Qt.MSWindowsFixedSizeDialogHint

  // Theme
  Material.theme: Material.Dark

  // StatusBar
  StatusBar{
    id: statusBar
  }

  // Loader
  Loader {
    id: mainLoader
    anchors {
      left: parent.left
      top: statusBar.bottom
      right: parent.right
      bottom: footer.top
    }
    source: "HomePageView.qml"
  }

  // Footer
  Rectangle {
    id: footer
    width: parent.width
    height: 18
    anchors.bottom: parent.bottom
    color: "black"
    Text {
      text: "Designed By Wei-Lun Chen (Ren) in TFRI"
      anchors.right: parent.right
      font.pixelSize: 14
      verticalAlignment: Text.AlignVCenter
      rightPadding: 8
      color: "lightgray"
    }
  }
}