using System.Collections.ObjectModel;
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

            cbFrom.SelectedIndex = 0;
            cbTo.SelectedIndex = 0;
        }

        private void HuggCast()
        {
            string HuggUrl = "huggingface.co";

            if(cbPath.Text.IndexOf(HuggUrl) != -1)
            {
                int Idx = cbPath.Text.IndexOf(HuggUrl) + HuggUrl.Length + 1;

                HuggUrl = cbPath.Text.Substring(Idx);
            }
            else
            {
                HuggUrl= cbPath.Text;
            }

            var AllPathces = HuggUrl.Split('/');
            HuggUrl = AllPathces[0] + "/" + AllPathces[1];

            string HuggFile = HuggUrl.Substring(0).Replace("/", "(slash)") + ".hgf";
            var Stream = System.IO.File.Create(FS.GetModelDir() + @"huggingface/" + HuggFile);
            Stream.Close();
        }

        private void OrigToDiff(bool bWait = false)
        {
            if (!System.IO.File.Exists(cbPath.Text))
                return;

            string SafeName = cbPath.Text;

            if(bWait)
            {
                CMD.ProcessConvertCKPT2Diff(SafeName);
            }
            else
            {
                Task.Run(() => CMD.ProcessConvertCKPT2Diff(SafeName));
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            int GetFromID = cbFrom.SelectedIndex; 
            int ToID = cbTo.SelectedIndex;

            // Anyway need cast to diff
            if (GetFromID == 0 || GetFromID == 1)
            {
                OrigToDiff(ToID == 1);
            }

            if (ToID == 1)
            {
                string Name = System.IO.Path.GetFileName(cbPath.Text);
                if(Name.EndsWith(".ckpt") || Name.EndsWith(".safetensors"))
                {
                    Name = System.IO.Path.GetFileNameWithoutExtension(cbPath.Text);
                }

                string SafeName = FS.GetModelDir() + "diff\\" + Name;
                Task.Run(() => CMD.ProcessConvertDiff2Onnx(SafeName));
            }
            else
            {
                if (GetFromID == 2)
                {
                    HuggCast();
                }
            }

            Helper.Form.UpdateModelsList();
        }

        private void cbTo_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            int ToID = cbTo.SelectedIndex;

            if (ToID == 1)
            {
                cbFrom.Items.Remove(cbFrom.Items[2]);
            }
            else
            {
                cbFrom.Items.Insert(2, "huggingface");
            }
        }
    }
}
