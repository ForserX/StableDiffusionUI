using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class HostFilter
    {
        public static void CheckTraceBack(string Message)
        {
            if (Message.Contains("Traceback (most recent call last)"))
            {
                Helper.UIHost.Dispatcher.Invoke(() => 
                { 
                    Helper.UIHost.Hide(); 
                    Helper.UIHost.Show(); 
                });

                // Drop loaded model
                Helper.Form.InvokeDropModel();

                Notification.SendErrorNotification("Error! See host for details!");
            }
        }
        public static void CheckImageSize(string Message)
        {
            if (Message.Contains("ValueError: `height` and `width` have to be divisible by 8"))
            {
                Helper.UIHost.Dispatcher.Invoke(() =>
                {
                    Helper.UIHost.Hide();
                });
                Notification.MsgBox("The image size is not a multiple of 8!");
            }
        }
        public static void CheckOutOfMemory(string Message)
        {
            if (Message.Contains("attention_probs = attention_scores.softmax(dim=-1)"))
            {
                Helper.UIHost.Dispatcher.Invoke(() =>
                {
                    Helper.UIHost.Hide();
                });
                Notification.MsgBox("CUDA GPU Error: Out of memory. Use fp16 or reduce the image size!");
            }

            if (Message.Contains("onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException"))
            {
                Helper.UIHost.Dispatcher.Invoke(() =>
                {
                    Helper.UIHost.Hide();
                });
                Notification.MsgBox("ONNX GPU Error: Out of memory. Use reduce the image size!");
            }
        }

        public static void CheckConvertState(string Message)
        {
            if (Message.Contains("SD: Done"))
            {
                Notification.SendNotification(Message);
                Helper.Form.InvokeUpdateModelsList();
            }

            if (Message.Contains("SD: TI Done"))
            {
                Notification.SendNotification("Textual inversion apply: done!");
                Helper.Form.InvokeUpdateModelsTIList();
            }

            if (Message.Contains("SD: Merge is done!"))
            {
                Notification.SendNotification("Models merge: done!");
                Helper.Form.InvokeUpdateModelsList();
            }
        }

        public static void CheckImageState(string Message)
        {
            if (Message.Contains("SD: Model loaded"))
            {
                Helper.Form.InvokeProgressUpdate(20);
            }

            if (Message.Contains("Model preload: done"))
            {
                Helper.Form.InvokeProgressUpdate(20);
            }

            if (Message.Contains("SD: Generating done"))
            {
                Helper.Form.InvokeProgressUpdate(60);
            }

            if (Message.Contains("SD Pipeline: Generating done!"))
            {
                Helper.Form.InvokeProgressUpdate(60);

                int UpSize = Helper.CurrentUpscaleSize;
                var Files = FS.GetFilesFrom(FS.GetWorkingDir(), new string[] { "png", "jpg" }, false);
                foreach (var file in Files)
                {
                    string NewFilePath = Helper.ImgPath + System.IO.Path.GetFileName(file);
                    System.IO.File.Move(file, NewFilePath);

                    Task.Run(() => CMD.UpscalerRunner(UpSize, NewFilePath));
                    if (UpSize == 0 || Helper.CurrentUpscalerType == Helper.UpscalerType.None)
                    {
                        Helper.Form.UpdateViewImg(NewFilePath);
                    }
                }

                Host.Print("\n  Task Done..... \n");
                Notification.SendNotification("Task: done!", true);
                Helper.Form.InvokeProgressUpdate(100);
                Helper.Form.UpdateCurrentViewImg();
            }

            if (Message.Contains("Image generate"))
            {
                Helper.Form.InvokeProgressApply();
            }
        }

        public static bool CheckFalseWarning(string Message)
        {
            if (Message == null || Message.Length == 0)
                return true;

            // DeepDanbooru check
            if (Message.Contains("rating:"))
                return false;
            
            // NSFW messages
            if (Message.Contains("safety check"))
                return false;

            if (Message.Contains(" warnings.warn("))
            {
                return true;
            }

            if (Message.Length > 310)
                return true;

            if (Helper.Mode != Helper.ImplementMode.Shark || Helper.Mode != Helper.ImplementMode.InvokeAI)
            {
                if (Message.Contains("Token indices sequence length is longer than the specified maximum sequence length for this model"))
                {
                    // Diffusers use long prompt extension
                    return true;
                }
            }

            return false;
        }

        public static void CheckControlNet(string Message)
        {
            if (Message.Contains("CN: Pose - "))
            {
                string PoseImg = Message.Replace("CN: Pose - ", string.Empty);
                Helper.Form.UpdateViewImg(PoseImg);
                Helper.Form.InvokeUpdateModelsList();
            }
            if (Message.Contains("CN: Model loaded"))
            {
                Helper.Form.InvokeProgressUpdate(20);
            }
            if (Message.Contains("CN: Image from Pose: done!"))
            {
                Helper.Form.InvokeProgressApply();
            }
        }
        public static void CheckDeepDanBooru(string Message)
        {
            if (Message.Contains("DeepDanBooru: Finished!"))
            {
                string NewPrompt = Message;
                NewPrompt = NewPrompt.Replace("\'", string.Empty);
                NewPrompt = NewPrompt.Replace("[", string.Empty);
                NewPrompt = NewPrompt.Replace("]", string.Empty);
                NewPrompt = NewPrompt.Replace("DeepDanBooru: Finished!", string.Empty);
                NewPrompt = NewPrompt.Replace("rating:safe", string.Empty);
                NewPrompt = NewPrompt.Replace("rating:questionable", string.Empty);
                NewPrompt = NewPrompt.Replace("rating:explicit", string.Empty);

                //NewPrompt = NewPrompt.Substring(1, NewPrompt.Length - 2);
                Helper.Form.InvokeSetPrompt(NewPrompt);
            }
        }
    }
}
