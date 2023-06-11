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

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для HuggDownload.xaml
    /// </summary>
    public partial class HuggDownload : Window
    {
        public HuggDownload()
        {
            InitializeComponent();
        }

        private void HuggCast()
        {
            string HuggUrl = "huggingface.co";

            if (tbUrl.Text.IndexOf(HuggUrl) != -1)
            {
                int Idx = tbUrl.Text.IndexOf(HuggUrl) + HuggUrl.Length + 1;

                HuggUrl = tbUrl.Text.Substring(Idx);
            }
            else
            {
                HuggUrl = tbUrl.Text;
            }

            var AllPathces = HuggUrl.Split('/');
            HuggUrl = AllPathces[0] + "/" + AllPathces[1];

            string HuggFile = HuggUrl.Substring(0).Replace("/", "(slash)") + ".hgf";
            var Stream = System.IO.File.Create(FS.GetModelDir() + @"huggingface/" + HuggFile);
            Stream.Close();
        }


        private void Button_Click(object sender, RoutedEventArgs e)
        {
            HuggCast();
            GlobalVariables.Form.InvokeUpdateModelsList();
            Close();
        }
    }
}
