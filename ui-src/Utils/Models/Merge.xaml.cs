using System;
using System.IO;
using System.Windows;

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для Merge.xaml
    /// </summary>
    public partial class Merge : Window
    {
        public Merge()
        {
            InitializeComponent();

            foreach(string Model in Directory.GetDirectories(FS.GetModelDir() + "diffusers"))
            {
                cbBase.Items.Add(Path.GetFileNameWithoutExtension(Model));
            }

            foreach (string Model in Directory.GetFiles(FS.GetModelDir() + "lora"))
            {
                cbLora.Items.Add(Path.GetFileNameWithoutExtension(Model));
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            if (tbOutpath.Text.Length == 0)
            {
                Notification.MsgBox("Set out file name!");
                return;
            }

            string InModel = $"\"{FS.GetModelDir() + "diffusers/" + cbBase.Text}\"";
            string AppendModel = $"\"{FS.GetModelDir() + "lora/" + cbLora.Text + ".safetensors"}\"";
            string OutDir = "\"" +FS.GetModelDir() + "diffusers/" + tbOutpath.Text + "\"";

            Host MergeCall = new Host(FS.GetWorkingDir());
            MergeCall.Start();
            MergeCall.Send("\"repo/" + PythonEnv.GetPy(Helper.VENV.Any) + "\" " + "\"repo/diffusion_scripts/merge_diff_lora.py\" " +
                    $" --model={InModel}" +
                    $" --lora_path={AppendModel}" +
                    $" --outpath={OutDir}"
                );

            //MergeCall.SendExitCommand();
            Helper.UIHost.Show();
        }
    }
}
