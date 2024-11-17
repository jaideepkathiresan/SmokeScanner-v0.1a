import eel
import wx
from py3.dialogs import *
from py3.imgsmoke import detectIMG
eel.init('web')

@eel.expose
def displayMessageBox(title, text, style):
	GenericMessageBox(title, text, style)
	return None

@eel.expose
def handleImage(wildcard="Images (*.jpeg,*.png)|*.jpeg;*.png"):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.STAY_ON_TOP
    dialog = wx.FileDialog(None, 'Select your Image file', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    test = detectIMG(path)
    result = test.procIMG()
    if result[0]:
        result[1] = result[1].decode("utf-8")
        return result[1]
    else:
        return "None"

eel.start('index.html', size=(976, 578))
#976*578