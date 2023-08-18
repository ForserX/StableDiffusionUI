using SD_FXUI.Utils;
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
            if (Message.Contains("Specified provider 'DmlExecutionProvider'"))
            {
                if (Notification.MsgBox("Warning! Incorrect DML Provider! To speed up the generation, you need to update the package. Update?"))
                {
                    // Drop loaded model
                    GlobalVariables.Form.InvokeDropModel();

                    HostReader.Filter("py -install onnxruntime-directml --force");
                }
            }

            if (Message.Contains("Traceback (most recent call last)"))
            {
                GlobalVariables.UIHost.Dispatcher.Invoke(() => 
                {
                    GlobalVariables.UIHost.Hide();
                    GlobalVariables.UIHost.Show(); 
                });

                // Drop loaded model
                GlobalVariables.Form.InvokeDropModel();

                Notification.SendErrorNotification("Error! See host for details!");
                GlobalVariables.LastVideoData.FrameDone = true;
            }
        }
        public static void CheckImageSize(string Message)
        {
            if (Message.Contains("ValueError: `height` and `width` have to be divisible by 8"))
            {
                GlobalVariables.UIHost.Dispatcher.Invoke(() =>
                {
                    GlobalVariables.UIHost.Hide();
                });
                Notification.MsgBox("The image size is not a multiple of 8!");
                GlobalVariables.LastVideoData.FrameDone = true;
            }
        }
        public static void CheckHalfPrecision(string Message)
        {
            if (Message.Contains("RuntimeError: \"LayerNormKernelImpl\" not implemented for 'Half'"))
            {
                GlobalVariables.UIHost.Dispatcher.Invoke(() =>
                {
                    GlobalVariables.UIHost.Hide();
                });
                Notification.MsgBox("Disable fp16. Not supported for current mode!");
                GlobalVariables.LastVideoData.FrameDone = true;
            }
        }

        public static void CheckOutOfMemory(string Message)
        {
            if (Message.Contains("attention_probs = attention_scores.softmax(dim=-1)"))
            {
                GlobalVariables.UIHost.Dispatcher.Invoke(() =>
                {
                    GlobalVariables.UIHost.Hide();
                });
                Notification.MsgBox("CUDA GPU Error: Out of memory. Use fp16 or reduce the image size!");
                GlobalVariables.LastVideoData.FrameDone = true;
            }

            if (Message.Contains("onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException"))
            {
                GlobalVariables.UIHost.Dispatcher.Invoke(() =>
                {
                    GlobalVariables.UIHost.Hide();
                });
                Notification.MsgBox("ONNX GPU Error: Out of memory. Use reduce the image size!");
                GlobalVariables.LastVideoData.FrameDone = true;
            }
        }

        public static void CheckConvertState(string Message)
        {
            if (Message.Contains("SD: Done"))
            {
                Notification.SendNotification(Message);
                GlobalVariables.Form.InvokeUpdateModelsList();
            }

            if (Message.Contains("SD: Merge is done!"))
            {
                Notification.SendNotification("Models merge: done!");
                GlobalVariables.Form.InvokeUpdateModelsList();
            }
        }

        public static void CheckImageState(string Message)
        {
            if (Message.Contains("SD: Model loaded"))
            {
                GlobalVariables.Form.InvokeProgressUpdate(20);
            }

            if (Message.Contains("Model preload: done"))
            {
                GlobalVariables.Form.InvokeProgressUpdate(20);
            }

            if (Message.Contains("SD: Generating done"))
            {
                GlobalVariables.Form.InvokeProgressUpdate(60);
            }

            if (Message.Contains("SD Pipeline: Generating done!"))
            {
                string CurrentVideoFrame = $"{GlobalVariables.LastVideoData.ActiveFrame}.png";

                GlobalVariables.LastVideoData.FrameDone = true;
                GlobalVariables.Form.InvokeProgressUpdate(60);

                int UpSize = GlobalVariables.CurrentUpscaleSize;
                var Files = FS.GetFilesFrom(FS.GetWorkingDir(), new string[] { "png", "jpg" }, false);
                foreach (var file in Files)
                {
                    string NewFilePath = GlobalVariables.ImgPath;
                    if (GlobalVariables.LastVideoData.ActiveRender)
                    {
                        NewFilePath += CurrentVideoFrame;
                    }
                    else
                    {
                        NewFilePath += System.IO.Path.GetFileName(file);
                    }

                    System.IO.File.Move(file, NewFilePath);

                    if (UpSize == 0 || GlobalVariables.CurrentUpscalerType == Helper.UpscalerType.None)
                    {
                        GlobalVariables.Form.UpdateViewImg(NewFilePath);
                    }
                    else
                    {
                        Task.Run(() => CMD.UpscalerRunner(UpSize, NewFilePath));
                        GlobalVariables.ImgList.Add(NewFilePath);
                    }
                }

                if (!GlobalVariables.LastVideoData.ActiveRender)
                {
                    Host.Print("\n  Task Done..... \n");
                    Notification.SendNotification("Task: done!", true);
                }

                GlobalVariables.Form.InvokeProgressUpdate(100);
                GlobalVariables.Form.UpdateCurrentViewImg();
            }

            if (Message.Contains("Image generate"))
            {
                GlobalVariables.Form.InvokeProgressApply();
            }
        }

        public static bool CheckFalseWarning(string Message)
        {
            if (Message == null || Message.Length == 0)
                return true;

            // SD Gen info check
            if (Message.Contains("Prompt:") || Message.Contains("Neg prompt:"))
                return false;

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

            if (Message.Length > 335)
                return true;

            if (GlobalVariables.Mode != Helper.ImplementMode.InvokeAI)
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
                GlobalVariables.Form.UpdateViewImg(PoseImg, true);
                GlobalVariables.Form.InvokeUpdateModelsList();
            }

            if (Message.Contains("CN: Model loaded"))
            {
                GlobalVariables.Form.InvokeProgressUpdate(20);
            }

            if (Message.Contains("CN: Image from Pose: done!"))
            {
                GlobalVariables.Form.InvokeProgressApply();
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
                GlobalVariables.Form.InvokeSetPrompt(NewPrompt);
            }
        }
    }
}
