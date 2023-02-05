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

namespace SD_FXUI.Utils
{
    /// <summary>
    /// Логика взаимодействия для HistoryList.xaml
    /// </summary>
    public partial class HistoryList : Window
    {
        public HistoryList()
        {
            InitializeComponent();

            foreach (var Itm in Helper.PromHistory)
                lbHistory.Items.Add(Itm);
        }

        private void ListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Helper.Form.SetPrompt(lbHistory.SelectedItem.ToString());
            this.Close();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            if(lbHistory.SelectedItem != null)
            {
                Helper.PromHistory.Remove(lbHistory.SelectedItem.ToString());
                lbHistory.Items.Remove(lbHistory.SelectedItem);
            }
        }
    }
}