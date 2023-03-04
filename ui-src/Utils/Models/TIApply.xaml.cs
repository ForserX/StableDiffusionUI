using System;
using System.IO;
using System.Windows;

namespace SD_FXUI.Utils.Models
{
    /// <summary>
    /// Логика взаимодействия для TIApply.xaml
    /// </summary>
    public partial class TIApply : Window
    {
        public TIApply()
        {
            InitializeComponent();

            foreach (string File in Directory.GetFiles(FS.GetModelDir() + "textual_inversion"))
            {
                string FixName = Path.GetFileNameWithoutExtension(File);
                cbTi.Items.Add(FixName);
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Helper.CurrentTI = cbTi.Text;
            Close();
        }
    }
}
