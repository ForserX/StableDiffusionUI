﻿using Microsoft.Toolkit.Uwp.Notifications;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    class Wrapper
    {
        public const int GWL_STYLE = -16;
        public const int WS_SYSMENU = 0x80000;
        public const int WS_CHILDWINDOW	= 0x40000;
        [DllImport("user32.dll", SetLastError = true)]
        public static extern int GetWindowLong(IntPtr hWnd, int nIndex);
        [DllImport("user32.dll")]
        public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

        public static void SendNotification(string Message = "Generation completed", string Caption = "Stable Diffusion: FX UI")
        {
            new ToastContentBuilder()
            .AddArgument("action", "viewConversation")
            .AddArgument("conversationId", 9813)
            .AddText(Caption)
            .AddText(Message)
            .Show();
        }

        public static bool MsgBox(string text, string caption = "Info")
        {
            return
                MessageBox.Show(text, caption, MessageBoxButton.OKCancel) == MessageBoxResult.OK;
        }
    }
}
