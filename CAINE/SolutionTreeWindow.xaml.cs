using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using CAINE.MachineLearning;

namespace CAINE
{
    /// <summary>
    /// Interactive troubleshooting window with decision tree navigation
    /// </summary>
    public partial class SolutionTreeWindow : Window
    {
        private InteractiveSolutionTree treeEngine;
        private InteractiveSolutionTree.DecisionNode currentNode;
        private InteractiveSolutionTree.DecisionNode rootNode;
        private InteractiveSolutionTree.TreePath currentPath;
        private string errorHash;
        private string errorText;
        private Stack<InteractiveSolutionTree.DecisionNode> navigationHistory;
        private ObservableCollection<TreeNodeViewModel> treeViewItems;

        /// <summary>
        /// View model for tree visualization
        /// </summary>
        public class TreeNodeViewModel : INotifyPropertyChanged
        {
            private string _icon;
            private string _question;
            private string _successRateText;
            private ObservableCollection<TreeNodeViewModel> _children;
            private bool _isExpanded;
            private bool _isSelected;

            public string Icon
            {
                get => _icon;
                set { _icon = value; OnPropertyChanged(nameof(Icon)); }
            }

            public string Question
            {
                get => _question;
                set { _question = value; OnPropertyChanged(nameof(Question)); }
            }

            public string SuccessRateText
            {
                get => _successRateText;
                set { _successRateText = value; OnPropertyChanged(nameof(SuccessRateText)); }
            }

            public ObservableCollection<TreeNodeViewModel> Children
            {
                get => _children ?? (_children = new ObservableCollection<TreeNodeViewModel>());
                set { _children = value; OnPropertyChanged(nameof(Children)); }
            }

            public bool IsExpanded
            {
                get => _isExpanded;
                set { _isExpanded = value; OnPropertyChanged(nameof(IsExpanded)); }
            }

            public bool IsSelected
            {
                get => _isSelected;
                set { _isSelected = value; OnPropertyChanged(nameof(IsSelected)); }
            }

            public event PropertyChangedEventHandler PropertyChanged;

            protected void OnPropertyChanged(string propertyName)
            {
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        public SolutionTreeWindow(string errorHash, string errorText)
        {
            InitializeComponent();
            this.errorHash = errorHash;
            this.errorText = errorText;
            treeEngine = new InteractiveSolutionTree();
            navigationHistory = new Stack<InteractiveSolutionTree.DecisionNode>();
            treeViewItems = new ObservableCollection<TreeNodeViewModel>();
            PathTreeView.ItemsSource = treeViewItems;

            _ = InitializeTreeAsync();
        }

        /// <summary>
        /// Initialize the decision tree
        /// </summary>
        private async Task InitializeTreeAsync()
        {
            try
            {
                CurrentQuestion.Text = "Loading decision tree...";
                ProgressBar.IsIndeterminate = true;

                // Build the tree
                rootNode = await treeEngine.BuildInteractiveTreeAsync(errorHash, errorText);
                currentNode = rootNode;
                currentPath = new InteractiveSolutionTree.TreePath { ErrorHash = errorHash };

                // Display first question
                DisplayCurrentNode();
                UpdateTreeVisualization();

                ProgressBar.IsIndeterminate = false;
                UpdateProgress(0);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to initialize tree: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Display the current node's question and actions
        /// </summary>
        private void DisplayCurrentNode()
        {
            if (currentNode == null) return;

            CurrentQuestion.Text = currentNode.Question;

            if (currentNode.IsLeaf)
            {
                // Show solution
                SolutionPanel.Visibility = Visibility.Visible;
                SolutionText.Text = currentNode.Solution;
                BtnYes.Visibility = Visibility.Collapsed;
                BtnNo.Visibility = Visibility.Collapsed;
                ShowFeedbackButtons();
            }
            else
            {
                // Show question and actions
                SolutionPanel.Visibility = Visibility.Collapsed;
                BtnYes.Visibility = Visibility.Visible;
                BtnNo.Visibility = Visibility.Visible;

                if (!string.IsNullOrEmpty(currentNode.ActionIfYes) || !string.IsNullOrEmpty(currentNode.ActionIfNo))
                {
                    ActionLabel.Visibility = Visibility.Visible;
                    SuggestedAction.Text = $"If Yes: {currentNode.ActionIfYes}\nIf No: {currentNode.ActionIfNo}";
                }
                else
                {
                    ActionLabel.Visibility = Visibility.Collapsed;
                    SuggestedAction.Text = "";
                }
            }

            // Update back button
            BtnBack.IsEnabled = navigationHistory.Count > 0;
        }

        /// <summary>
        /// Handle Yes button click
        /// </summary>
        private void BtnYes_Click(object sender, RoutedEventArgs e)
        {
            if (currentNode?.YesChild != null)
            {
                NavigateToNode(currentNode.YesChild, true);
            }
        }

        /// <summary>
        /// Handle No button click
        /// </summary>
        private void BtnNo_Click(object sender, RoutedEventArgs e)
        {
            if (currentNode?.NoChild != null)
            {
                NavigateToNode(currentNode.NoChild, false);
            }
        }

        /// <summary>
        /// Navigate to a new node
        /// </summary>
        private void NavigateToNode(InteractiveSolutionTree.DecisionNode newNode, bool decision)
        {
            // Add to history
            navigationHistory.Push(currentNode);
            currentPath.NodesVisited.Add(currentNode);
            currentPath.Decisions.Add(decision);

            // Move to new node
            currentNode = newNode;
            DisplayCurrentNode();
            UpdateTreeVisualization();
            UpdateProgress((double)currentPath.NodesVisited.Count / 5); // Assume max 5 steps
        }

        /// <summary>
        /// Handle Back button click
        /// </summary>
        private void BtnBack_Click(object sender, RoutedEventArgs e)
        {
            if (navigationHistory.Count > 0)
            {
                currentNode = navigationHistory.Pop();

                // Remove from path
                if (currentPath.NodesVisited.Count > 0)
                {
                    currentPath.NodesVisited.RemoveAt(currentPath.NodesVisited.Count - 1);
                    currentPath.Decisions.RemoveAt(currentPath.Decisions.Count - 1);
                }

                DisplayCurrentNode();
                UpdateTreeVisualization();
                UpdateProgress((double)currentPath.NodesVisited.Count / 5);
            }
        }

        /// <summary>
        /// Handle Restart button click
        /// </summary>
        private void BtnRestart_Click(object sender, RoutedEventArgs e)
        {
            currentNode = rootNode;
            navigationHistory.Clear();
            currentPath = new InteractiveSolutionTree.TreePath { ErrorHash = errorHash };
            DisplayCurrentNode();
            UpdateTreeVisualization();
            UpdateProgress(0);
            HideFeedbackButtons();
        }

        /// <summary>
        /// Update the tree visualization
        /// </summary>
        private void UpdateTreeVisualization()
        {
            treeViewItems.Clear();

            if (currentPath.NodesVisited.Count == 0 && currentNode != null)
            {
                // Show current node as root
                var rootViewModel = CreateTreeNodeViewModel(currentNode, "❓");
                treeViewItems.Add(rootViewModel);
            }
            else
            {
                // Show path taken
                TreeNodeViewModel parent = null;
                for (int i = 0; i < currentPath.NodesVisited.Count; i++)
                {
                    var node = currentPath.NodesVisited[i];
                    var decision = i < currentPath.Decisions.Count ? currentPath.Decisions[i] : false;
                    var icon = decision ? "✓" : "✗";

                    var nodeViewModel = CreateTreeNodeViewModel(node, icon);

                    if (parent == null)
                    {
                        treeViewItems.Add(nodeViewModel);
                    }
                    else
                    {
                        parent.Children.Add(nodeViewModel);
                    }

                    parent = nodeViewModel;
                    parent.IsExpanded = true;
                }

                // Add current node
                if (parent != null && currentNode != null)
                {
                    var currentViewModel = CreateTreeNodeViewModel(currentNode, "→");
                    currentViewModel.IsSelected = true;
                    parent.Children.Add(currentViewModel);
                }
            }
        }

        /// <summary>
        /// Create tree node view model
        /// </summary>
        private TreeNodeViewModel CreateTreeNodeViewModel(InteractiveSolutionTree.DecisionNode node, string icon)
        {
            return new TreeNodeViewModel
            {
                Icon = icon,
                Question = node.IsLeaf ? "Solution" : node.Question.Length > 50
                    ? node.Question.Substring(0, 47) + "..." : node.Question,
                SuccessRateText = $"({node.SuccessRate:P0})"
            };
        }

        /// <summary>
        /// Update progress bar
        /// </summary>
        private void UpdateProgress(double value)
        {
            ProgressBar.Value = Math.Min(100, value * 100);
            ProgressText.Text = $"Step {currentPath.NodesVisited.Count} of estimated 5";
        }

        /// <summary>
        /// Show feedback buttons
        /// </summary>
        private void ShowFeedbackButtons()
        {
            FeedbackLabel.Visibility = Visibility.Visible;
            BtnFeedbackYes.Visibility = Visibility.Visible;
            BtnFeedbackNo.Visibility = Visibility.Visible;
        }

        /// <summary>
        /// Hide feedback buttons
        /// </summary>
        private void HideFeedbackButtons()
        {
            FeedbackLabel.Visibility = Visibility.Collapsed;
            BtnFeedbackYes.Visibility = Visibility.Collapsed;
            BtnFeedbackNo.Visibility = Visibility.Collapsed;
        }

        /// <summary>
        /// Handle positive feedback
        /// </summary>
        private async void BtnFeedbackYes_Click(object sender, RoutedEventArgs e)
        {
            currentPath.WasSuccessful = true;
            currentPath.EndTime = DateTime.Now;
            await treeEngine.UpdateTreeWeightsAsync(currentPath, true);

            MessageBox.Show("Thank you for your feedback! The solution path has been marked as successful.",
                "Feedback Recorded", MessageBoxButton.OK, MessageBoxImage.Information);

            HideFeedbackButtons();
        }

        /// <summary>
        /// Handle negative feedback
        /// </summary>
        private async void BtnFeedbackNo_Click(object sender, RoutedEventArgs e)
        {
            currentPath.WasSuccessful = false;
            currentPath.EndTime = DateTime.Now;
            await treeEngine.UpdateTreeWeightsAsync(currentPath, false);

            MessageBox.Show("Thank you for your feedback. This path will be deprioritized in future troubleshooting.",
                "Feedback Recorded", MessageBoxButton.OK, MessageBoxImage.Information);

            HideFeedbackButtons();
        }

        /// <summary>
        /// Show full tree visualization
        /// </summary>
        private void BtnShowFullTree_Click(object sender, RoutedEventArgs e)
        {
            // This would open a separate window with full tree visualization
            MessageBox.Show("Full tree visualization coming in next version!", "Feature Preview",
                MessageBoxButton.OK, MessageBoxImage.Information);
        }

        /// <summary>
        /// Close window
        /// </summary>
        private void BtnClose_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}