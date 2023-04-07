using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace SD_FXUI.Utils.Models
{
    /// <summary>
    /// Логика взаимодействия для ModelSelector.xaml
    /// </summary>
    public partial class ModelSelector : Window
    {
        public ModelSelector()
        {
            InitializeComponent();
        }

        private void Button_ClickUpdate(object sender, RoutedEventArgs e)
        {
            imgView.Children.Clear();

            int imageWidth = 64;
            int imageHeight = 64;
            int offset = 0;

            int kW = (int)(imgView.ActualWidth / imageWidth);
            int kH = (int)((imgView.ActualHeight - offset) / imageHeight);

            for (int i = 0; i < kW; i++)
            {
                for(int j = 0; j < kH; j++)
                {
                    Image newImage = new Image();
                    //newCheckBox.Content = tokensAllInMemory[i];
                    newImage.Name = "Image" + i;
                    newImage.Height = imageHeight;
                    newImage.Width = imageWidth;
                    newImage.HorizontalAlignment = HorizontalAlignment.Left;
                    newImage.VerticalAlignment = VerticalAlignment.Top;
                    newImage.Source = Helper.getImageSourceUndefined();             

                    newImage.Margin = new Thickness(imageWidth * i, imageHeight * j, 0.0, 0.0);
                    imgView.Children.Add(newImage);
                }
                

            }

        }
    }
}
