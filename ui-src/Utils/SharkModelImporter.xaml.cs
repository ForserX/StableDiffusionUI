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
    /// Логика взаимодействия для SharkModelImporter.xaml
    /// </summary>
    public partial class SharkModelImporter : Window
    {
        public SharkModelImporter()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            string HuggUrl = "huggingface.co";
            int Idx = cbPath.Text.IndexOf(HuggUrl) + HuggUrl.Length;

            HuggUrl = cbPath.Text.Substring(Idx);

            var AllPathces = HuggUrl.Split('/');
            HuggUrl += AllPathces[0] + AllPathces[1];

            Helper.ModelsList.Add(HuggUrl.Substring(1));
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            string DiffPath = FS.GetModelDir() + "diff\\";

            string Name = System.IO.Path.GetFileName(cbPath.Text);
            
            if(System.IO.File.Exists(DiffPath + Name))
            {
                FS.CopyDirectory(cbPath.Text, DiffPath + Name, true);
            }

            Helper.ModelsList.Add(Name);
        }
    }
}
