﻿using System;
using System.Collections.Generic;
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
using System.Windows.Shapes;
using Microsoft.Win32;
using Windows.Storage.Pickers;
using WinRT;
using Microsoft.WindowsAPICodePack.Dialogs;

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для Settings.xaml
    /// </summary>
    public partial class Settings : Window
    {
        static public bool UseNotif = false;
        static public bool UseNotifImgs = false;
        static public bool UseInternalVAE = false;

        bool IsLoadedWnd = false;
        public Settings()
        {
            InitializeComponent();
            chNotification.IsChecked = UseNotif;
            chNotification_1.IsChecked = UseNotifImgs;
            tsVAE.IsChecked = UseInternalVAE;

            tbCustomModelPath.Text = FS.GetModelDir();

            IsLoadedWnd = true;
        }

        private void chNotification_Checked(object sender, RoutedEventArgs e)
        {
            if (IsLoadedWnd)
            {
                UseNotif = chNotification.IsChecked.Value;
            }
        }

        private void chNotification_1_Checked(object sender, RoutedEventArgs e)
        {
            if (IsLoadedWnd)
            {
                UseNotifImgs = chNotification_1.IsChecked.Value;
            }
        }

        private void chVAE_Checked(object sender, RoutedEventArgs e)
        {
            if (IsLoadedWnd)
            {
                UseInternalVAE = tsVAE.IsChecked.Value;
            }

            GlobalVariables.Form.InvokeUpdateModelsList();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            tbCustomModelPath.Text = FS.GetWorkingDir() + "/models/";
        }

        private async void Button_Click_1(object sender, RoutedEventArgs e)
        {
            var dialog = new CommonOpenFileDialog();
            dialog.IsFolderPicker = true;
            CommonFileDialogResult result = dialog.ShowDialog();

            if (result == CommonFileDialogResult.Ok && !string.IsNullOrWhiteSpace(dialog.FileName))
            {
                tbCustomModelPath.Text = dialog.FileName;
            }

        }

        private void tbCustomModelPath_TextChanged(object sender, TextChangedEventArgs e)
        {
            GlobalVariables.ModelsDir = tbCustomModelPath.Text.Replace('\\', '/');

            if (!GlobalVariables.ModelsDir.EndsWith('/'))
            {
                GlobalVariables.ModelsDir += "/";
            }
        }
    }
}
