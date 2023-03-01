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

                Notification.SendErrorNotification("Error! See host for details!");
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
        }

        public static void CheckImageState(string Message)
        {
            if (Message.Contains("SD: Model loaded"))
            {
                Helper.Form.InvokeProgressUpdate(20);
            }

            if (Message.Contains("SD: Generating done"))
            {
                Helper.Form.InvokeProgressUpdate(60);
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

            if (Message.Length > 300)
                return true;


            if (Message.Contains(" warnings.warn("))
            {
                return true;
            }

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

        public static string FixString(string Message)
        {
            string NewMessage = Message;

            return NewMessage;
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
