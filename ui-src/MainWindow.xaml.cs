using System;
using System.Collections.Generic;
using System.Configuration;
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
using System.Windows.Navigation;
using System.Windows.Shapes;
using static System.Net.Mime.MediaTypeNames;

namespace SD_FXUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public static string CachePath = string.Empty;
        public static string ImgPath = string.Empty;
        public static MainWindow Form = null;
        public static HostForm UIHost = null;
        Configuration Config = null;
        public MainWindow()
        {
            InitializeComponent();

            CachePath = FS.GetModelDir() + @"\shark\";
            ImgPath = FS.GetWorkingDir() + "\\images\\" + System.DateTime.Now.ToString().Replace(':', '-').Replace(' ', '_') + "\\";
            ImgPath.Replace('\\', '/');

            System.IO.Directory.CreateDirectory(CachePath);
            System.IO.Directory.CreateDirectory(ImgPath);

            cbUpscaler.SelectedIndex = 0;
            cbModel.SelectedIndex = 0;
            cbX.SelectedIndex = 3;
            cbY.SelectedIndex = 3;
            Form = this;

            UIHost = new HostForm();
            UIHost.Hide();

            // Load App data
            Config = ConfigurationManager.OpenExeConfiguration(ConfigurationUserLevel.None);
            Load();
        }
        void Load()
        {
            if (Config.AppSettings.Settings["fp16"] == null)
            {
                return;
            }

            cbFf16.IsChecked = Config.AppSettings.Settings["fp16"].Value == "true";
            cbY.Text = Config.AppSettings.Settings["height"].Value;
            cbX.Text = Config.AppSettings.Settings["width"].Value;
            NegPrompt.Text = Config.AppSettings.Settings["neg"].Value;
            tbSteps.Text = Config.AppSettings.Settings["steps"].Value;

            var ListModel = Config.AppSettings.Settings["model"].Value.Split('|');

            foreach (var model in ListModel)
            {
                cbModel.Items.Add(model);
            }
        }
        void Save()
        {
            Config.AppSettings.Settings.Add("fp16", cbFf16.IsChecked == true ? "true" : "false");
            Config.AppSettings.Settings.Add("height", cbY.Text);
            Config.AppSettings.Settings.Add("width", cbX.Text);
            Config.AppSettings.Settings.Add("neg", NegPrompt.Text);
            Config.AppSettings.Settings.Add("steps", tbSteps.Text);

            string Models = "";

            foreach (var a in cbModel.Items)
            {
                Models += a.ToString() + "|";
            }

            Config.AppSettings.Settings.Add("model", Models);

            Config.Save(ConfigurationSaveMode.Full, true);
            ConfigurationManager.RefreshSection("appSettings");
        }
        private string GetCommandLine()
        {
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string CmdLine = "\"../../repo/shark.venv/Scripts/python.exe\" ../../repo/stable_diffusion/scripts/txt2img.py ";
            CmdLine += $"--precision={FpMode} --device=vulkan" + $" --prompt=\"{TryPrompt.Text}\" --negative_prompts=\"{NegPrompt.Text}\" ";
            CmdLine += $"--height={cbY.Text} --width={cbX.Text} ";
            CmdLine += $"--guidance_scale={tbCFG.Text.Replace(',', '.')} ";
            CmdLine += $" --steps={tbSteps.Text} --seed={tbSeed.Text} ";
            CmdLine += $"--hf_model_id=\"{cbModel.Text}\" ";
            //           CmdLine += "--model_variant=\"D:\\Neirotrash\\StableDiffusionUI-Shark-AMD\\data\\models\\anything-v4.0\" -max_length=77 --version=\"v1_4\" ";
            CmdLine += "--no-use_tuned --local_tank_cache=\".//\" ";

            CmdLine += "--enable_stack_trace";

            // CmdLine += $" --output_dir=\"{ImgPath}\" ";

            return CmdLine;
        }
        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            string cmdline = GetCommandLine();
            Helper.UpscalerType Type = (Helper.UpscalerType)cbUpscaler.SelectedIndex;
            int Size = (int)slUpscale.Value;
            int TotalSize = int.Parse(tbTotalCount.Text);

            await Task.Run(() => CMD.ProcessRunner(cmdline, TotalSize, Type, Size));
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbSteps != null)
                tbSteps.Text = slSteps.Value.ToString();
        }

        private void tbSteps2_TextChanged(object sender, TextChangedEventArgs e)
        {
            double Val = 0;
            double.TryParse(tbCFG.Text, out Val);
            slCFG.Value = Val;
        }

        private void Slider2_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbCFG != null)
                tbCFG.Text = slCFG.Value.ToString();
        }

        private void slUpscale_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (lbUpscale != null)
                lbUpscale.Name = "x" + (slUpscale.Value + 1).ToString();
        }

        private void tbSteps_TextChanged(object sender, TextChangedEventArgs e)
        {
            double Val = 0;
            double.TryParse(tbSteps.Text, out Val);
            slSteps.Value = Val;
        }

        void SetImg(string Img)
        {
            ViewImg.Source = new System.Windows.Media.Imaging.BitmapImage(new Uri(Img));
        }
        public void UpdateViewImg(string Img)
        {
            Dispatcher.Invoke(() => SetImg(Img));
        }

        private void btFolder_ValueChanged(object sender, MouseButtonEventArgs e)
        {
            string argument = "/select, \"" + ImgPath + "\"";
            System.Diagnostics.Process.Start("explorer.exe", argument);
        }
        private void btCmd_ValueChanged(object sender, MouseButtonEventArgs e)
        {
            UIHost.Hide();
            UIHost.Show();
        }

        private void OnClose(object sender, EventArgs e)
        {
            UIHost.Close();
            Save();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            Utils.SharkModelImporter Importer = new Utils.SharkModelImporter();
            Importer.ShowDialog();

            cbModel.Items.Clear();

            foreach (var Itm in Helper.ModelsList)
            {
                cbModel.Items.Add(Itm);
            }

            cbModel.SelectedIndex = cbModel.Items.Count- 1;
        }
    }
}
