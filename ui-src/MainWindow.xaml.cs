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
        Config Data = null;
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

            cbSampler.SelectedIndex = 0;
            cbDevice.SelectedIndex = 0;

            Helper.Form = this;

            Helper.UIHost = new HostForm();
            Helper.UIHost.Hide();

            // Load App data
            Data = new Config();
            Load();
        }
        void Load()
        {
            cbFf16.IsChecked = Data.Get("fp16") == "true";
            cbY.Text =  Data.Get("height");
            cbX.Text = Data.Get("width");
            NegPrompt.Text =  Data.Get("neg");
            tbSteps.Text =  Data.Get("steps");
            cbUpscaler.Text =  Data.Get("upscaler");
            cbDevice.Text =  Data.Get("device");
            cbSampler.Text =  Data.Get("sampler");
            var ListModel =  Data.Get("model").Split('|');

            foreach (var model in ListModel)
            {
                if(model.Length > 0)
                {
                    cbModel.Items.Add(model);
                }
            }
        }
        void Save()
        {
             Data.Set("fp16", cbFf16.IsChecked == true ? "true" : "false");
             Data.Set("height", cbY.Text);
             Data.Set("width", cbX.Text);
             Data.Set("neg", NegPrompt.Text);
             Data.Set("steps", tbSteps.Text);
             Data.Set("upscaler", cbUpscaler.Text);
             Data.Set("sampler", cbSampler.Text);
             Data.Set("device", cbDevice.Text);

            string Models = "";

            foreach (var a in cbModel.Items)
            {
                Models += a.ToString() + "|";
            }

            Data.Set("model", Models);

            Data.Save();
        }
        private string GetCommandLine()
        {
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string Model = cbModel.Text.IndexOf('/') != -1 ? cbModel.Text : FS.GetModelDir() + "diff\\" + cbModel.Text;
            string CmdLine = $" --precision={FpMode}" 
                    + $" --device=\"{cbDevice.Text}\""
                    + $" --prompt=\"{TryPrompt.Text}\"" 
                    + $" --negative_prompts=\"{NegPrompt.Text}\""
                    + $" --height={cbY.Text}"
                    + $" --width={cbX.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scheduler={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --total_count={tbTotalCount.Text}"
                    + $" --hf_model_id=\"{Model}\""
                    + $" --no-use_tuned " 
                    + $" --local_tank_cache=\".//\""
                    +  " --enable_stack_trace" 
                    +  " --write_metadata_to_png"
            ;

            return CmdLine;
        }
        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            if(chRandom.IsChecked.Value)
            {
                var rand = new Random();
                tbSeed.Text = rand.Next().ToString();
            }
               
        

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
