using HandyControl.Tools.Extension;
using Microsoft.Toolkit.Uwp.Notifications;
using System;
using System.Reflection;
using System.Windows;

namespace SD_FXUI
{
    internal class Notification
    {
        public static void SendNotification(string Message = "Generation completed", bool ApplyImages = false, string Caption = "Stable Diffusion: XUI")
        {
            if (!Utils.Settings.UseNotif)
                return;

            ToastContentBuilder NotificationToShow = new ToastContentBuilder();

            //NotificationToShow.AddAppLogoOverride(new System.Uri(
            //    Assembly.GetExecutingAssembly().GetName().Name +
            //    ";component/res/icon.png", UriKind.RelativeOrAbsolute), 
            //    
            //    ToastGenericAppLogoCrop.Circle);
            NotificationToShow.AddArgument("action", "viewConversation");
            NotificationToShow.AddArgument("conversationId", 1435);
            NotificationToShow.AddText(Caption);
            NotificationToShow.AddText(Message);

            if (ApplyImages && Utils.Settings.UseNotifImgs)
            {
                foreach (string File in Helper.ImgList)
                {
                    if (Helper.ImgList.Count == 1)
                    {
                        NotificationToShow.AddHeroImage(new Uri(File));
                        break;
                    }

                    NotificationToShow.AddInlineImage(new Uri(File));
                }
            }

            try
            {
                NotificationToShow.Show();
            }
            catch
            {
                Host.Print("Toast show error!");
            }

        }
        public static void SendErrorNotification(string Message = "Generation completed", string Caption = "Stable Diffusion: XUI")
        {
            if (!Utils.Settings.UseNotif)
                return;

            ToastContentBuilder NotificationToShow = new ToastContentBuilder();

            NotificationToShow.AddArgument("action", "viewConversation");
            NotificationToShow.AddArgument("conversationId", 1435);
            NotificationToShow.AddText(Caption);
            NotificationToShow.AddText(Message);

            NotificationToShow.AddButton(new ToastButton()
                .SetContent("Open host")
                .AddArgument("action", "OpenHost")
                .SetBackgroundActivation());

            NotificationToShow.Show();
        }

        public static void ToastBtnClickManager(ToastNotificationActivatedEventArgsCompat Arg)
        {
            if (Arg.Argument.Contains("OpenHost"))
            {
                Helper.UIHost.Dispatcher.Invoke(() =>
                {
                    Helper.UIHost.Hide();
                    Helper.UIHost.Show();
                });
            }
        }

        public static bool MsgBox(string text, string caption = "Info")
        {
            return
                MessageBox.Show(text, caption, MessageBoxButton.OKCancel) == MessageBoxResult.OK;
        }
    }
}
