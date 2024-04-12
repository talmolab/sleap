from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QKeySequence
import sys

class Commands:
    def __init__(self):
        # simulate instance to group assignments for simplicity
        self.instance_groups = {}

    def assignInstanceToGroup(self, instance_id, group):
        # assign the instance to the selected group
        self.instance_groups[instance_id] = group
        print(f"Assigned Instance {instance_id} to Group {group}")

class CommandContext:
    def __init__(self, commands):
        self.commands = commands
        self.current_instance = "Instance1"  # placeholder for currently selected instance

    def executeAssignToGroup(self, group):
        self.commands.assignInstanceToGroup(self.current_instance, group)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.commands = Commands()
        self.command_context = CommandContext(self.commands)
        self.initUI()

    def initUI(self):
        menuBar = self.menuBar()
        tracksMenu = menuBar.addMenu("Tracks")
        self.addInstanceGroupAssignmentMenu(menuBar)
        self.setWindowTitle('Main Application Window')
        self.setGeometry(300, 300, 250, 150)
        self.show()

    def addInstanceGroupAssignmentMenu(self, menuBar):
        instanceGroupMenu = menuBar.addMenu("Assign Instance to Group")
        instanceGroupMenu.setToolTipsVisible(True)

        # example groups
        instance_groups = ["Group A", "Group B", "Group C"]
        # hotkeys for each group assignment
        hotkeys = ["Ctrl+Shift+A", "Ctrl+Shift+B", "Ctrl+Shift+C"]
        
        for group, hotkey in zip(instance_groups, hotkeys):
            action = QAction(f"Assign to {group}", instanceGroupMenu)
            action.setShortcut(QKeySequence(hotkey))
            # using CommandContext to execute the command
            action.triggered.connect(lambda checked, g=group: self.command_context.executeAssignToGroup(g))
            instanceGroupMenu.addAction(action)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
