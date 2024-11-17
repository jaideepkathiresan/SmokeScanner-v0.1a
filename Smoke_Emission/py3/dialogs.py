from win32api import MessageBox
from win32con import MB_ICONSTOP, MB_ICONWARNING, MB_ICONINFORMATION

def GenericMessageBox(title, text, style):
    if style==0:
        MessageBox(0, text, title, MB_ICONSTOP)
    elif style==1:
        MessageBox(0, text, title, MB_ICONWARNING)
    elif style==2:
        MessageBox(0, text, title, MB_ICONINFORMATION)
    else:
        pass
    return None