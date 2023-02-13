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

            return message.StartsWith("Downloading") || Char.IsNumber(Firt);
        }
        void ImplPrint(string message)
        {
            if (StepTest(message))
            {
                string OldText = tbHost.Text.Substring(0, tbHost.Text.Length - 2);
                int Idx = OldText.LastIndexOf("\n");

                if (Idx != -1 && StepTest(OldText.Substring(Idx + 1)))
                {
                    tbHost.Text = OldText.Substring(0, Idx + 1) + message + "\n";
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
            if (message.Length > 300)
                return;

            System.Diagnostics.Trace.WriteLine(message);
            message = message.Replace(FS.GetWorkingDir(), "${Workspace}");

            Dispatcher.Invoke(() => ImplPrint(message));
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
    }
}
