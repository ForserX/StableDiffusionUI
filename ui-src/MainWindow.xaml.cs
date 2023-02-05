using HandyControl.Controls;
using HandyControl.Data;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using static System.Net.Mime.MediaTypeNames;

namespace SD_FXUI
{
    /// <summary>
    /// Interaction logic for BlurWindow.xaml
    /// </summary>
    public partial class MainWindow : HandyControl.Controls.BlurWindow
    {
        List<string> ImgList = new List<string>();
        Config Data = null;
        ImageSource NoImageData = null;

        public MainWindow()
        {
            InitializeComponent();

            Helper.CachePath = FS.GetModelDir() + @"\shark\";
            Helper.ImgPath = FS.GetWorkingDir() + "\\images\\" + System.DateTime.Now.ToString().Replace(':', '-').Replace(' ', '_') + "\\";
            Helper.ImgPath.Replace('\\', '/');
            
            System.IO.Directory.CreateDirectory(Helper.CachePath);
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\huggingface");
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\onnx");
            System.IO.Directory.CreateDirectory(FS.GetModelDir() + @"\diff");

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
            ChangeTheme();

            gridImg.Visibility = Visibility.Collapsed;
            NoImageData = ViewImg.Source;
        }
        void Load()
        {
            cbFf16.IsChecked = Data.Get("fp16") == "true";
            cbY.Text = Data.Get("height");
            cbX.Text = Data.Get("width");
            NegPrompt.Text = Data.Get("neg");
            tbSteps.Text = Data.Get("steps");
            cbUpscaler.Text = Data.Get("upscaler");
            cbSampler.Text = Data.Get("sampler");
            var ListModel = Data.Get("model");

            switch (Data.Get("back_mode"))
            {
                case "0": Helper.Mode = Helper.ImplementMode.InvokeAI; break;
                case "1": btnShark_Click(0, new RoutedEventArgs()); break;
                case "2": btnONNX_Click(0, new RoutedEventArgs()); break;

                default: btnShark_Click(0, new RoutedEventArgs()); break;
            }

            UpdateModelsList();
            cbModel.Text = ListModel.ToString();
            cbDevice.Text = Data.Get("device");

            var HistoryStack = Data.Get("history").Split('|');
            foreach (var item in HistoryStack)
            {
                if (item.Length > 0)
                    Helper.PromHistory.Add(item);
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

            Data.Set("model", cbModel.Text);
            Data.Set("back_mode", ((int)(Helper.Mode)).ToString());

            string HistoryStack = "";
            foreach (var item in Helper.PromHistory)
            {
                HistoryStack += item + "|";
            }
            Data.Set("history", HistoryStack);

            // Save to file
            Data.Save();
        }

        private string GetCommandLineOnnx()
        {
            string Model = FS.GetModelDir() + "onnx\\" + cbModel.Text;
            string CmdLine = $""
                    + $" --prompt=\"{TryPrompt.Text}\""
                    + $" --prompt_neg=\"{NegPrompt.Text}\""
                    + $" --height={cbY.Text}"
                    + $" --width={cbX.Text}"
                    + $" --guidance_scale={tbCFG.Text.Replace(',', '.')}"
                    + $" --scmode={cbSampler.Text}"
                    + $" --steps={tbSteps.Text}"
                    + $" --seed={tbSeed.Text}"
                    + $" --totalcount={tbTotalCount.Text}"
                    + $" --model=\"{Model}\""
                    + $" --outpath=\"{FS.GetWorkingDir()}\""
            ;

            if (Helper.DrawMode == Helper.DrawingMode.Img2Img)
            {
                CmdLine += $" --mode=\"img2img\" --img=\"{Helper.InputImagePath}\" --imgscale=0.{tbDenoising.Text}";
            }

            return CmdLine;
        }
        private string GetCommandLineShark()
        {
            string FpMode = cbFf16.IsChecked.Value ? "fp16" : "fp32";
            string Model = cbModel.Text.IndexOf('/') != -1 ? cbModel.Text : FS.GetModelDir() + "diff\\" + cbModel.Text;
            string CmdLine = $" --precision={FpMode}" 
                    + $" --device=\"{cbDevice.Text}\""
                    + $" --prompt=\"{TryPrompt.Text}\"" 
                    + $" --negative-prompts=\"{NegPrompt.Text}\""
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
//                    + " --iree-vulkan-target-triple=rdna3-unknown-windows"
                    +  " --write_metadata_to_png"
            ;

            return CmdLine;
        }

        public void SetPrompt(string Prompt) => TryPrompt.Text = Prompt;

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            System.IO.Directory.CreateDirectory(Helper.ImgPath);

            if (chRandom.IsChecked.Value)
            {
                var rand = new Random();
                tbSeed.Text = rand.Next().ToString();
            }

            Helper.UpscalerType Type = (Helper.UpscalerType)cbUpscaler.SelectedIndex;
            int Size = (int)slUpscale.Value;

            string cmdline = "";
            switch (Helper.Mode)
            {
                case Helper.ImplementMode.Shark:
                    {
                        cmdline += GetCommandLineShark();
                        Task.Run(() => CMD.ProcessRunnerShark(cmdline, Type, Size));
                        break;
                    }
                case Helper.ImplementMode.ONNX:
                    {
                        cmdline += GetCommandLineOnnx();
                        Task.Run(() => CMD.ProcessRunnerOnnx(cmdline, Type, Size));
                        break;
                    }
            }

            if (Helper.PromHistory.Count == 0 || Helper.PromHistory[Helper.PromHistory.Count- 1] != TryPrompt.Text)
            {
                Helper.PromHistory.Add(TryPrompt.Text);
            }

            ClearImages();
        }

        private void Slider_Denoising(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (tbDenoising != null)
                tbDenoising.Text = slDenoising.Value.ToString();
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
            ViewImg.Source = new BitmapImage(new Uri(Img));
            ListViewItemsCollections.Add(new ListViewItemsData()
            {
                GridViewColumnName_ImageSource = Img
            });

            ListView1.ItemsSource = ListViewItemsCollections;
            ImgList.Add(Img);
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
            Importer.Show();
        }

        public void UpdateModelsList()
        {
            cbModel.Items.Clear();

            foreach (var Itm in FS.GetModels(Helper.Mode))
            {
                cbModel.Items.Add(Itm);
            }

            cbModel.SelectedIndex = cbModel.Items.Count - 1;
        }

        private void chRandom_Checked(object sender, RoutedEventArgs e)
        {
            if (tbSeed != null)
                tbSeed.IsEnabled = false;
        }
        private void chRandom_Unchecked(object sender, RoutedEventArgs e)
        {
            tbSeed.IsEnabled = true;
        }

        private void cbDevice_TextChanged(object sender, RoutedEventArgs e)
        {

        }

        private void btnONNX_Click(object sender, RoutedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.ONNX)
            {
                Helper.Mode = Helper.ImplementMode.ONNX;

                var Safe = btnONNX.Background;
                btnONNX.Background = new SolidColorBrush(Colors.DarkOrchid);
                btnShark.Background = Safe;

                UpdateModelsList();

                cbDevice.Items.Clear();
                cbDevice.Items.Add("GPU: 0");

                // #TODO: GPU List check
                cbDevice.Items.Add("GPU: 1");

                btImg.Visibility = Visibility.Visible;
                cbFf16.Visibility = Visibility.Hidden;
            }
        }

        private void btnShark_Click(object sender, RoutedEventArgs e)
        {
            if (Helper.Mode != Helper.ImplementMode.Shark)
            {
                Helper.Mode = Helper.ImplementMode.Shark;

                var Safe = btnShark.Background;
                btnShark.Background = new SolidColorBrush(Colors.DarkMagenta);
                btnONNX.Background = Safe;

                UpdateModelsList();
                cbDevice.Items.Clear();
                cbDevice.Items.Add("vulkan");
                cbDevice.Items.Add("CUDA");

                btImg.Visibility = Visibility.Hidden;
                cbFf16.Visibility = Visibility.Visible;
            }
        }

        public ObservableCollection<ListViewItemsData> ListViewItemsCollections { get { return _ListViewItemsCollections; } }
        ObservableCollection<ListViewItemsData> _ListViewItemsCollections = new ObservableCollection<ListViewItemsData>();
        public class ListViewItemsData
        {
            public string GridViewColumnName_ImageSource { get; set; }
            public string GridViewColumnName_ID { get; set; }
        }

        private void ListView1_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if(ImgList.Count> 0)
            {
                ViewImg.Source = new BitmapImage(new Uri(ImgList[ListView1.SelectedIndex]));
            }
        }
        void ClearImages()
        {
            ImgList.Clear();
            ViewImg.Source = NoImageData;

            ListView1.UnselectAll();
            ListView1.ItemsSource = null;
            ListView1.Items.Clear();
            ListViewItemsCollections.Clear();
        }

        public void InvokeClearImages()
        {
            Dispatcher.Invoke(() => { InvokeClearImages(); });
        }

        private void ChangeTheme()
        {            
            {
                Resources.MergedDictionaries.Clear();
                Resources.MergedDictionaries.Add(new ResourceDictionary
                {
                    Source = new Uri($"pack://application:,,,/HandyControl;component/Themes/Skin{SkinType.Violet.ToString()}.xaml")
                });
                Resources.MergedDictionaries.Add(new ResourceDictionary
                {
                    Source = new Uri("pack://application:,,,/HandyControl;component/Themes/Theme.xaml")
                });
            }
        }

        private void cbDevice_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (cbDevice.SelectedItem == null)
                return;

            if (cbDevice.SelectedItem.ToString() == "GPU: 1" || cbDevice.SelectedItem.ToString() == "GPU: 0")
            {
                string FileName = FS.GetWorkingDir() + @"\repo\shark.venv\Lib\site-packages\diffusers\pipelines\onnx_utils.py";

                if(!System.IO.File.Exists(FileName))
                {
                    return;
                }
                using (var reader = System.IO.File.OpenText(FileName))
                {
                    int LineCounter = 0;
                    string? str = reader.ReadLine();
                    while (str != null)
                    {
                        if (str.Contains("InferenceSession"))
                        {
                            if (cbDevice.SelectedItem.ToString() == "GPU: 1")
                            {
                                str = "        return ort.InferenceSession(path, providers=[provider], provider_options=[{'device_id': 1}], sess_options=sess_options)";
                            }
                            else
                            {
                                str = "        return ort.InferenceSession(path, providers=[provider], provider_options=[{'device_id': 0}], sess_options=sess_options)";
                            }
                            break;
                        }
                        str = reader.ReadLine();
                        LineCounter++;
                    }

                    reader.Close();

                    string[] Lines = System.IO.File.ReadAllLines(FileName);
                    Lines[LineCounter] = str;
                    System.IO.File.WriteAllLines(FileName, Lines);
                }
            }
        }

        private void btImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog OpenDlg = new OpenFileDialog();
            OpenDlg.Filter = "PNG (*.png)|*.png|JPG (*.jpg)|*.jpg|All files (*.*)|*.*";
            OpenDlg.Multiselect = false;

            bool? IsOpened = OpenDlg.ShowDialog();
            if (IsOpened.Value)
            {
                Helper.InputImagePath = OpenDlg.FileName;
                gridImg.Visibility = Visibility.Visible;
                imgLoaded.Source = new BitmapImage(new Uri(Helper.InputImagePath));
                Helper.DrawMode = Helper.DrawingMode.Img2Img;
            }
        }

        private void tbDenoising_TextChanged(object sender, TextChangedEventArgs e)
        {
            slDenoising.Value = float.Parse(tbDenoising.Text.Replace('.', ','));
        }

        private void btImageClear_Click(object sender, RoutedEventArgs e)
        {
            gridImg.Visibility = Visibility.Collapsed;

            Helper.DrawMode = Helper.DrawingMode.Text2Img;
            imgLoaded.Source = NoImageData;
        }

        private void btHistory_Click(object sender, MouseButtonEventArgs e)
        {
            Utils.HistoryList HistoryWnd = new Utils.HistoryList();
            HistoryWnd.ShowDialog();
        }

        private void BlurWindow_Loaded(object sender, RoutedEventArgs e)
        {
            Install.Check();
        }
    }
}
