; AutoHotkey v2 Script
; Emergency Python Process Killer
; Hotkey: Ctrl + Shift + Alt + P
; Kills all python.exe processes immediately

^+!p::  ; Ctrl + Shift + Alt + P
{
    ; Show confirmation message
    MsgBox("Killing all Python processes...", "Emergency Python Killer", "T1")
    
    ; Kill all python.exe processes
    Run("taskkill /F /IM python.exe /T", , "Hide")
    
    ; Also kill pythonw.exe (background Python)
    Run("taskkill /F /IM pythonw.exe /T", , "Hide")
    
    ; Brief delay to let processes terminate
    Sleep(500)
    
    ; Show completion message
    TrayTip("Python Killer", "All Python processes terminated!", 16)
    
    return
}

; Optional: Show startup message
TrayTip("Python Killer Active", "Press Ctrl+Shift+Alt+P to kill Python", 16)
