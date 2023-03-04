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
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Helper.Mode = Helper.ImplementMode.DiffCUDA;
            this.Close();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            Helper.Mode = Helper.ImplementMode.ONNX;
            this.Close();
        }

        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            Helper.Mode = Helper.ImplementMode.Shark;
            this.Close();
        }
    }
}
