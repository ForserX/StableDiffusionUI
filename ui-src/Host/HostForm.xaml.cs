﻿using SD_FXUI.Debug;
using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;

namespace SD_FXUI
{
    /// <summary>
    /// Логика взаимодействия для HostForm.xaml
    /// </summary>
    public partial class HostForm : Window
    {
        public HostForm()
        {
            InitializeComponent();
        }

        bool StepTest(string message)
        {
            if(message == null || message.Length == 0)
            { 
                return false; 
            }

            char Firt = message[0];
            int ChrId = 0;
            foreach (char symbol in message)
            {
                if (symbol == ' ' || symbol == '\t')
                {
                    Firt = message[ChrId + 1];
                }
                else
                {
                    break;
                }
                ChrId++;
            }

            return message.StartsWith("DownloadProgress") || message.StartsWith("Downloading") || char.IsNumber(Firt);
        }

        public void Clear()
        {
            tbHost.Text = "";
        }

        void ImplPrint(string message)
        {
            // Buffer offload
            if (tbHost.Text.Length > 8000)
            {
                tbHost.Text = tbHost.Text[500..];
            }
                
            if (StepTest(message))
            {
                string OldText = tbHost.Text[..^2];
                int Idx = OldText.LastIndexOf("\n");

                if (Idx != -1 && StepTest(OldText[(Idx + 1)..]))
                {
                    tbHost.Text = OldText[..(Idx + 1)] + message + "\n";
                }
                else
                {
                    tbHost.Text += message + "\n";
                }
            }
            else
            {
                tbHost.Text += message + "\n";
            }
            tbHost.SelectionStart = tbHost.Text.Length - 1;
            tbHost.SelectionLength = 0;
            tbHost.ScrollToEnd();
        }

        public void Print(string message)
        {
            System.Diagnostics.Trace.WriteLine(message);
            message = message.Replace(FS.GetWorkingDir(), "${Workspace}");

            Dispatcher.Invoke(() => ImplPrint(message));

            Log.SendMessageToFileFromHost(message);
        }

        private void OnClosing(object sender, EventArgs e)
        {
        }

        private void OnActiveted(object sender, RoutedEventArgs e)
        {
           var hwnd = new WindowInteropHelper(this).Handle;
           Wrapper.SetWindowLong(hwnd, Wrapper.GWL_STYLE, Wrapper.GetWindowLong(hwnd, Wrapper.GWL_STYLE) & ~Wrapper.WS_SYSMENU);
        }

        private void OnClosing(object sender, MouseButtonEventArgs e)
        {
            Hide();
        }

        private void tbCmd_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                HostReader.Filter(tbCmd.Text);
                tbCmd.Text = "";
            }
        }
    }
}
