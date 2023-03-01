using System;
using System.Windows;
using System.Windows.Input;

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для ImageView.xaml
    /// </summary>
    public partial class ImageView : Window
    {
        public ImageView()
        {
            InitializeComponent();
        }

        public void SetImage(string Path)
        {
            imgView.Source = FS.BitmapFromUri(new Uri(Path));
            //this.Height = imgView.Source.Height;
            //this.Width = imgView.Source.Width;
        }

        private void KeyDownEvent(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                Close();
            }
        }
    }
}
