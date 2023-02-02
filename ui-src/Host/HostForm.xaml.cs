using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

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
            message = Regex.Replace(message, @"\s+", "");

            return Char.IsNumber(message[0]);
        }
        void ImplPrint(string message)
        {
            if (StepTest(message))
            {
                string OldText = tbHost.Text.Substring(0, tbHost.Text.Length - 1);
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
            tbHost.SelectionStart = tbHost.Text.Length;
            tbHost.SelectionLength = 0;

            tbHost.ScrollToEnd();
        }

        public void Print(string message)
        {
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
