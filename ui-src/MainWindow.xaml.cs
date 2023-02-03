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
        Configuration Config = null;
        public MainWindow()
        {
            InitializeComponent();

            Helper.CachePath = FS.GetModelDir() + @"\shark\";
            Helper.ImgPath = FS.GetWorkingDir() + "\\images\\" + System.DateTime.Now.ToString().Replace(':', '-').Replace(' ', '_') + "\\";
            Helper.ImgPath.Replace('\\', '/');
            
            System.IO.Directory.CreateDirectory(Helper.CachePath);
            System.IO.Directory.CreateDirectory(Helper.ImgPath);

            cbUpscaler.SelectedIndex = 0;
            cbModel.SelectedIndex = 0;
            cbX.SelectedIndex = 3;
            cbY.SelectedIndex = 3;
            Helper.Form = this;

            Helper.UIHost = new HostForm();
            Helper.UIHost.Hide();

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
            cbUpscaler.Text = Config.AppSettings.Settings["upscaler"].Value;
            cbDevice.Text = Config.AppSettings.Settings["device"].Value;
            var ListModel = Config.AppSettings.Settings["model"].Value.Split('|');

            foreach (var model in ListModel)
            {
                cbModel.Items.Add(model);
            }
        }
        void Save()
        {
            Config.AppSettings.Settings["fp16"].Value = cbFf16.IsChecked == true ? "true" : "false";
            Config.AppSettings.Settings["height"].Value = cbY.Text;
            Config.AppSettings.Settings["width"].Value = cbX.Text;
            Config.AppSettings.Settings["neg"].Value = NegPrompt.Text;
            Config.AppSettings.Settings["steps"].Value = tbSteps.Text;
            Config.AppSettings.Settings["upscaler"].Value = cbUpscaler.Text;
            Config.AppSettings.Settings["device"].Value = cbDevice.Text;

            string Models = "";

            foreach (var a in cbModel.Items)
            {
                Models += a.ToString() + "|";
            }

            Config.AppSettings.Settings["model"].Value = Models;

            Config.Save(ConfigurationSaveMode.Full, true);
            ConfigurationManager.RefreshSection("appSettings");
        }
        private string GetCommandLine()
        {
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string Model = cbModel.Text.IndexOf('/') != -1 ? cbModel.Text : FS.GetModelDir() + "diff\\" + cbModel.Text;
            string CmdLine = "\"../../repo/shark.venv/Scripts/python.exe\" ../../repo/stable_diffusion/scripts/txt2img.py ";
            CmdLine += $"--precision={FpMode} --device=\"{cbDevice.Text}\"" + $" --prompt=\"{TryPrompt.Text}\" --negative_prompts=\"{NegPrompt.Text}\" ";
            CmdLine += $"--height={cbY.Text} --width={cbX.Text} ";
            CmdLine += $"--guidance_scale={tbCFG.Text.Replace(',', '.')} -scheduler=\"PNDM\"";
            CmdLine += $" --steps={tbSteps.Text} --seed={tbSeed.Text} --total_count={tbTotalCount.Text} ";
            CmdLine += $"--hf_model_id=\"{Model}\" ";

            CmdLine += "--no-use_tuned --local_tank_cache=\".//\" ";
            CmdLine += "--enable_stack_trace --write_metadata_to_png";

            return CmdLine;
        }
        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            var rand = new Random();
            tbSeed.Text = rand.Next().ToString();

            string cmdline = GetCommandLine();
            Helper.UpscalerType Type = (Helper.UpscalerType)cbUpscaler.SelectedIndex;
            int Size = (int)slUpscale.Value;

            await Task.Run(() => CMD.ProcessRunner(cmdline, Type, Size));
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
                lbUpscale.Content = "x" + (slUpscale.Value + 1).ToString();
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
            string argument = "/select, \"" + Helper.ImgPath + "\"";
            System.Diagnostics.Process.Start("explorer.exe", argument);
        }
        private void btCmd_ValueChanged(object sender, MouseButtonEventArgs e)
        {
            Helper.UIHost.Hide();
            Helper.UIHost.Show();
        }

        private void OnClose(object sender, EventArgs e)
        {
            Helper.UIHost.Close();
            Save();
        }

        private void Button_ClickBreak(object sender, RoutedEventArgs e)
        {
            foreach(var Proc in Helper.SecondaryProcessList)
            {
                Proc.Kill();
            }

            Helper.UIHost.Print("\n All task aborted (」°ロ°)」");
            Helper.SecondaryProcessList.Clear();
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

        private void chRandom_Checked(object sender, RoutedEventArgs e)
        {
            tbSeed.IsEnabled = false;
        }
        private void chRandom_Unchecked(object sender, RoutedEventArgs e)
        {
            tbSeed.IsEnabled = true;
        }

        private void cbDevice_TextChanged(object sender, RoutedEventArgs e)
        {

        }
    }
}
