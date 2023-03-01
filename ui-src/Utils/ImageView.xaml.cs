using System;
using System.Windows;

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
    }
}
