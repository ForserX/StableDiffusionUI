using System.Threading.Tasks;
using System.Windows;

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
            int Idx = cbPath.Text.IndexOf(HuggUrl) + HuggUrl.Length + 1;

            HuggUrl = cbPath.Text.Substring(Idx);

            var AllPathces = HuggUrl.Split('/');
            HuggUrl = AllPathces[0] + "/" + AllPathces[1];

            string HuggFile = HuggUrl.Substring(0).Replace("/", "(slash)") + ".hgf";
            var Stream = System.IO.File.Create(FS.GetModelDir() + @"huggingface/" + HuggFile);
            Stream.Close();

            Helper.Form.UpdateModelsList();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            string DiffPath = FS.GetModelDir() + "diff\\";

            string Name = System.IO.Path.GetFileName(cbPath.Text);

            if (System.IO.File.Exists(DiffPath + Name) && System.IO.File.Exists(cbPath.Text))
            {
                FS.CopyDirectory(cbPath.Text, DiffPath + Name, true);
            }

            Helper.Form.UpdateModelsList();
        }

        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            if (!System.IO.File.Exists(cbPath.Text))
                return;

            string SafeName = cbPath.Text;
            Task.Run(() => CMD.ProcessConvertCKPT2Diff(SafeName));

            Helper.Form.UpdateModelsList();
        }
    }
}
