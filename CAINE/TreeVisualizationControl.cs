using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using CAINE.MachineLearning;

namespace CAINE.Controls
{
    /// <summary>
    /// Custom control for rendering decision tree as a visual flowchart
    /// </summary>
    public class TreeVisualizationControl : Canvas
    {
        private InteractiveSolutionTree.DecisionNode rootNode;
        private Dictionary<string, NodeVisual> nodeVisuals = new Dictionary<string, NodeVisual>();
        private double nodeWidth = 150;
        private double nodeHeight = 60;
        private double horizontalSpacing = 180;
        private double verticalSpacing = 100;

        /// <summary>
        /// Visual representation of a tree node
        /// </summary>
        private class NodeVisual
        {
            public Border Container { get; set; }
            public TextBlock Text { get; set; }
            public TextBlock SuccessRate { get; set; }
            public Point Position { get; set; }
            public InteractiveSolutionTree.DecisionNode Node { get; set; }
            public List<Line> Connections { get; set; } = new List<Line>();
        }

        /// <summary>
        /// Render the tree
        /// </summary>
        public void RenderTree(InteractiveSolutionTree.DecisionNode root)
        {
            this.Children.Clear();
            nodeVisuals.Clear();
            rootNode = root;

            if (root == null) return;

            // Calculate tree layout
            var layout = CalculateTreeLayout(root);

            // Create visual nodes
            CreateNodeVisuals(root, layout);

            // Draw connections
            DrawConnections();

            // Adjust canvas size
            AdjustCanvasSize();
        }

        /// <summary>
        /// Calculate positions for all nodes
        /// </summary>
        private Dictionary<string, Point> CalculateTreeLayout(InteractiveSolutionTree.DecisionNode node)
        {
            var layout = new Dictionary<string, Point>();
            var levelNodes = new Dictionary<int, List<InteractiveSolutionTree.DecisionNode>>();

            // Build level structure
            BuildLevelStructure(node, 0, levelNodes);

            // Calculate positions
            double currentY = 50;
            foreach (var level in levelNodes)
            {
                int nodeCount = level.Value.Count;
                double totalWidth = nodeCount * nodeWidth + (nodeCount - 1) * (horizontalSpacing - nodeWidth);
                double startX = (this.ActualWidth > 0 ? this.ActualWidth : 800) / 2 - totalWidth / 2;

                for (int i = 0; i < level.Value.Count; i++)
                {
                    var n = level.Value[i];
                    double x = startX + i * horizontalSpacing;
                    layout[n.NodeId] = new Point(x, currentY);
                }

                currentY += verticalSpacing;
            }

            return layout;
        }

        /// <summary>
        /// Build level structure for tree layout
        /// </summary>
        private void BuildLevelStructure(InteractiveSolutionTree.DecisionNode node, int level,
            Dictionary<int, List<InteractiveSolutionTree.DecisionNode>> levelNodes)
        {
            if (node == null) return;

            if (!levelNodes.ContainsKey(level))
                levelNodes[level] = new List<InteractiveSolutionTree.DecisionNode>();

            levelNodes[level].Add(node);

            BuildLevelStructure(node.YesChild, level + 1, levelNodes);
            BuildLevelStructure(node.NoChild, level + 1, levelNodes);
        }

        /// <summary>
        /// Create visual elements for nodes
        /// </summary>
        private void CreateNodeVisuals(InteractiveSolutionTree.DecisionNode node, Dictionary<string, Point> layout)
        {
            if (node == null || nodeVisuals.ContainsKey(node.NodeId)) return;

            var position = layout.ContainsKey(node.NodeId) ? layout[node.NodeId] : new Point(0, 0);

            // Create container
            var border = new Border
            {
                Width = nodeWidth,
                Height = nodeHeight,
                CornerRadius = new CornerRadius(5),
                BorderThickness = new Thickness(2)
            };

            // Set colors based on node type and success rate
            if (node.IsLeaf)
            {
                border.Background = new SolidColorBrush(Color.FromRgb(45, 90, 45)); // Green for solutions
                border.BorderBrush = new SolidColorBrush(Color.FromRgb(60, 120, 60));
            }
            else if (node.SuccessRate > 0.7)
            {
                border.Background = new SolidColorBrush(Color.FromRgb(45, 45, 90)); // Blue for high success
                border.BorderBrush = new SolidColorBrush(Color.FromRgb(60, 60, 120));
            }
            else if (node.SuccessRate < 0.3)
            {
                border.Background = new SolidColorBrush(Color.FromRgb(90, 45, 45)); // Red for low success
                border.BorderBrush = new SolidColorBrush(Color.FromRgb(120, 60, 60));
            }
            else
            {
                border.Background = new SolidColorBrush(Color.FromRgb(60, 60, 60)); // Gray for neutral
                border.BorderBrush = new SolidColorBrush(Color.FromRgb(80, 80, 80));
            }

            // Create text elements
            var stackPanel = new StackPanel
            {
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(5)
            };

            var questionText = new TextBlock
            {
                Text = TruncateText(node.IsLeaf ? "Solution" : node.Question, 50),
                Foreground = Brushes.White,
                TextWrapping = TextWrapping.Wrap,
                FontSize = 11,
                HorizontalAlignment = HorizontalAlignment.Center
            };

            var successText = new TextBlock
            {
                Text = $"{node.SuccessRate:P0} ({node.TimesUsed} uses)",
                Foreground = new SolidColorBrush(Color.FromRgb(200, 200, 200)),
                FontSize = 9,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 2, 0, 0)
            };

            stackPanel.Children.Add(questionText);
            stackPanel.Children.Add(successText);
            border.Child = stackPanel;

            // Position the node
            Canvas.SetLeft(border, position.X);
            Canvas.SetTop(border, position.Y);

            // Add to canvas
            this.Children.Add(border);

            // Store visual reference
            var nodeVisual = new NodeVisual
            {
                Container = border,
                Text = questionText,
                SuccessRate = successText,
                Position = position,
                Node = node
            };
            nodeVisuals[node.NodeId] = nodeVisual;

            // Add hover effect
            border.MouseEnter += (s, e) =>
            {
                border.BorderBrush = Brushes.Yellow;
                ShowNodeTooltip(node, position);
            };

            border.MouseLeave += (s, e) =>
            {
                border.BorderBrush = node.IsLeaf ?
                    new SolidColorBrush(Color.FromRgb(60, 120, 60)) :
                    new SolidColorBrush(Color.FromRgb(80, 80, 80));
                HideNodeTooltip();
            };

            // Recursively create children
            CreateNodeVisuals(node.YesChild, layout);
            CreateNodeVisuals(node.NoChild, layout);
        }

        /// <summary>
        /// Draw connections between nodes
        /// </summary>
        private void DrawConnections()
        {
            foreach (var visual in nodeVisuals.Values)
            {
                var node = visual.Node;

                if (node.YesChild != null && nodeVisuals.ContainsKey(node.YesChild.NodeId))
                {
                    DrawConnection(visual, nodeVisuals[node.YesChild.NodeId], true);
                }

                if (node.NoChild != null && nodeVisuals.ContainsKey(node.NoChild.NodeId))
                {
                    DrawConnection(visual, nodeVisuals[node.NoChild.NodeId], false);
                }
            }
        }

        /// <summary>
        /// Draw a single connection line
        /// </summary>
        private void DrawConnection(NodeVisual from, NodeVisual to, bool isYes)
        {
            var line = new Line
            {
                X1 = from.Position.X + nodeWidth / 2,
                Y1 = from.Position.Y + nodeHeight,
                X2 = to.Position.X + nodeWidth / 2,
                Y2 = to.Position.Y,
                Stroke = isYes ? Brushes.LightGreen : Brushes.LightCoral,
                StrokeThickness = 2,
                StrokeDashArray = isYes ? null : new DoubleCollection { 5, 3 }
            };

            // Add arrow
            var arrow = CreateArrow(
                new Point(line.X2, line.Y2),
                new Point(line.X1, line.Y1),
                isYes ? Brushes.LightGreen : Brushes.LightCoral
            );

            // Add label
            var label = new TextBlock
            {
                Text = isYes ? "Yes" : "No",
                Foreground = isYes ? Brushes.LightGreen : Brushes.LightCoral,
                FontSize = 10,
                FontWeight = FontWeights.Bold
            };

            Canvas.SetLeft(label, (line.X1 + line.X2) / 2 - 10);
            Canvas.SetTop(label, (line.Y1 + line.Y2) / 2 - 10);

            // Insert behind nodes
            this.Children.Insert(0, line);
            this.Children.Insert(0, arrow);
            this.Children.Add(label);

            from.Connections.Add(line);
        }

        /// <summary>
        /// Create arrow polygon
        /// </summary>
        private Polygon CreateArrow(Point tip, Point from, Brush color)
        {
            var arrow = new Polygon
            {
                Fill = color,
                Points = new PointCollection()
            };

            // Calculate arrow points
            double angle = Math.Atan2(tip.Y - from.Y, tip.X - from.X);
            double arrowLength = 10;
            double arrowAngle = Math.PI / 6;

            arrow.Points.Add(tip);
            arrow.Points.Add(new Point(
                tip.X - arrowLength * Math.Cos(angle - arrowAngle),
                tip.Y - arrowLength * Math.Sin(angle - arrowAngle)
            ));
            arrow.Points.Add(new Point(
                tip.X - arrowLength * Math.Cos(angle + arrowAngle),
                tip.Y - arrowLength * Math.Sin(angle + arrowAngle)
            ));

            return arrow;
        }

        /// <summary>
        /// Show tooltip for node
        /// </summary>
        private ToolTip currentTooltip;
        private void ShowNodeTooltip(InteractiveSolutionTree.DecisionNode node, Point position)
        {
            currentTooltip = new ToolTip
            {
                Content = new StackPanel
                {
                    Children =
                    {
                        new TextBlock { Text = node.Question, FontWeight = FontWeights.Bold },
                        new TextBlock { Text = $"Success Rate: {node.SuccessRate:P0}" },
                        new TextBlock { Text = $"Times Used: {node.TimesUsed}" },
                        new TextBlock { Text = $"Category: {node.ErrorCategory}" }
                    }
                },
                Background = new SolidColorBrush(Color.FromRgb(40, 40, 40)),
                Foreground = Brushes.White,
                BorderBrush = Brushes.Gray,
                BorderThickness = new Thickness(1)
            };

            currentTooltip.IsOpen = true;
        }

        /// <summary>
        /// Hide tooltip
        /// </summary>
        private void HideNodeTooltip()
        {
            if (currentTooltip != null)
            {
                currentTooltip.IsOpen = false;
                currentTooltip = null;
            }
        }

        /// <summary>
        /// Truncate text for display
        /// </summary>
        private string TruncateText(string text, int maxLength)
        {
            if (string.IsNullOrEmpty(text)) return "";
            return text.Length <= maxLength ? text : text.Substring(0, maxLength - 3) + "...";
        }

        /// <summary>
        /// Adjust canvas size to fit all nodes
        /// </summary>
        private void AdjustCanvasSize()
        {
            double maxX = 0, maxY = 0;
            foreach (var visual in nodeVisuals.Values)
            {
                maxX = Math.Max(maxX, visual.Position.X + nodeWidth);
                maxY = Math.Max(maxY, visual.Position.Y + nodeHeight);
            }

            this.Width = maxX + 50;
            this.Height = maxY + 50;
        }

        /// <summary>
        /// Highlight a path through the tree
        /// </summary>
        public void HighlightPath(List<InteractiveSolutionTree.DecisionNode> path)
        {
            // Reset all nodes to default
            foreach (var visual in nodeVisuals.Values)
            {
                visual.Container.BorderThickness = new Thickness(2);
            }

            // Highlight path nodes
            foreach (var node in path)
            {
                if (nodeVisuals.ContainsKey(node.NodeId))
                {
                    nodeVisuals[node.NodeId].Container.BorderThickness = new Thickness(4);
                    nodeVisuals[node.NodeId].Container.BorderBrush = Brushes.Gold;
                }
            }
        }
    }
}