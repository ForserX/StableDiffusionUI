using System.Windows;

namespace SD_FXUI
{
    /// <summary>
    /// Логика взаимодействия для Welcome.xaml
    /// </summary>
    public partial class Welcome : Window
    {
        public Welcome()
        {
            InitializeComponent();
            cbPyVer.SelectedIndex = 0;
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            GlobalVariables.Mode = Helper.ImplementMode.DiffCUDA;
            GlobalVariables.PythonVersion = cbPyVer.Text;
            this.Close();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            GlobalVariables.Mode = Helper.ImplementMode.ONNX;
            GlobalVariables.PythonVersion = cbPyVer.Text;
            this.Close();
        }
    }
}
