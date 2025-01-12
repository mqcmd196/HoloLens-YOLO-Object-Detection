using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.Robotics.ROSTCPConnector;
using Unity.Sentis;
using UnityEngine;
using CompressedImageMsg = RosMessageTypes.Sensor.CompressedImageMsg;
using SegmentationInfoMsg = RosMessageTypes.DeticRos.SegmentationInfoMsg;
using UnityEngine.UI;

namespace Assets.Scripts
{
    /// <summary>
    ///     Model executor using a YOLO object detection model.
    /// </summary>
    public class YoloModelExecutor : MonoBehaviour
    {
        /// <summary>
        ///     Object detection model that should be executed.
        /// </summary>
        public ModelAsset ModelAsset;

        /// <summary>
        ///     Material with shader that is used for scaling the input image to the correct aspect ratio.
        /// </summary>
        public Material ShaderForScaling;

        private TextureTransform textureTransform;

        private bool disposed;

        private IWorker worker;

        private TensorFloat inputTensor;

        private TensorFloat outputTensor;

        private ModelState modelState = ModelState.PreProcessing;

        private CameraTransform cameraTransform;

        private bool hasMoreModelToRun = true;

        private IEnumerator modelEnumerator;

        private RenderTexture intermediateRenderTexture;

        private YoloDebugOutput yoloDebugOutput;

        private int layerCount;

        private float threshold;

        private YoloRecognitionHandler yoloRecognitionHandler;

        private static WebCamTextureAccess WebCamTextureAccess => WebCamTextureAccess.Instance;

        private static SettingsProvider SettingsProvider => SettingsProvider.Instance;

        public struct BoundingBox2D
        {
            public int minX;
            public int minY;
            public int maxX;
            public int maxY;
        }

        private ROSConnection rosConnection;

        private byte[] jpegData = null;

        [SerializeField] private RawImage debugRawImage;

        private void Start()
        {
            // Get other components
            this.yoloDebugOutput = gameObject.GetComponent<YoloDebugOutput>();
            this.yoloRecognitionHandler = gameObject.GetComponent<YoloRecognitionHandler>();

            // Initialize settings
            this.SettingsProviderOnPropertyChanged(null,
                new PropertyChangedEventArgs(nameof(SettingsProvider.ModelExecutionOR)));
            this.SettingsProviderOnPropertyChanged(null,
                new PropertyChangedEventArgs(nameof(SettingsProvider.ThresholdOR)));
            SettingsProvider.PropertyChanged += this.SettingsProviderOnPropertyChanged;

            // Load the model from the provided NNModel asset
            Model model = ModelLoader.Load(this.ModelAsset);

            // Create a Barracuda worker to run the model on the GPU
            this.worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

            // Initialize model input
            WebCamTextureAccess.Play();
            // this.intermediateRenderTexture =
            //     new RenderTexture(Parameters.ModelImageResolution.x, Parameters.ModelImageResolution.y, 24);
            this.intermediateRenderTexture = new RenderTexture(WebCamTextureAccess.ActualCameraSize.x, WebCamTextureAccess.ActualCameraSize.y, 24);
            // this.ShaderForScaling.SetFloat("_Aspect",
            //     (float)WebCamTextureAccess.ActualCameraSize.x / WebCamTextureAccess.ActualCameraSize.y * Parameters.ModelImageResolution.y / Parameters.ModelImageResolution.x);
            // this.textureTransform = new TextureTransform().SetDimensions(Parameters.ModelImageResolution.x, Parameters.ModelImageResolution.y, 3);

            // Initialize ROS
            rosConnection = ROSConnection.GetOrCreateInstance();
            rosConnection.RosIPAddress = "133.11.216.96";
            rosConnection.RosPort = 10000;
            rosConnection.RegisterPublisher<CompressedImageMsg>("ar/image/compressed");
            rosConnection.Subscribe<SegmentationInfoMsg>("ar/segmentation_info", Callback);
        }

        private void Update()
        {
            this.cameraTransform = new CameraTransform(Camera.main);
            // Graphics.Blit(WebCamTextureAccess.WebCamTexture, this.intermediateRenderTexture, this.ShaderForScaling);
            Graphics.Blit(WebCamTextureAccess.WebCamTexture, this.intermediateRenderTexture);
            RenderTexture.active = this.intermediateRenderTexture;
            Texture2D texture2D = new Texture2D(this.intermediateRenderTexture.width,
                this.intermediateRenderTexture.height, TextureFormat.RGB24, false);
            texture2D.ReadPixels(
                new Rect(0, 0, this.intermediateRenderTexture.width, this.intermediateRenderTexture.height), 0, 0);
            texture2D.Apply();

            ShowCapturedTexture(texture2D);

            RenderTexture.active = null;
            jpegData = texture2D.EncodeToJPG();
            CompressedImageMsg rosImageData = new CompressedImageMsg
            {
                format = "jpeg",
                data = jpegData,
            };
            rosConnection.Send("ar/image/compressed", rosImageData);

            // Check subscription

            // switch (this.modelState)
            // {
            //     case ModelState.Idle:
            //         break;
            //     case ModelState.PreProcessing:
            //         this.inputTensor?.Dispose();
            //         this.cameraTransform = new CameraTransform(Camera.main);
            //         Graphics.Blit(WebCamTextureAccess.WebCamTexture, this.intermediateRenderTexture,
            //             this.ShaderForScaling);
            //         RenderTexture.active = this.intermediateRenderTexture;
            //         Texture2D texture2D = new Texture2D(this.intermediateRenderTexture.width,
            //             this.intermediateRenderTexture.height, TextureFormat.RGB24, false);
            //         texture2D.ReadPixels(
            //             new Rect(0, 0, this.intermediateRenderTexture.width, this.intermediateRenderTexture.height), 0,
            //             0);
            //         texture2D.Apply();
            //         RenderTexture.active = null;
            //         // this.inputTensor = TextureConverter.ToTensor(this.intermediateRenderTexture, this.textureTransform);
            //         this.modelState = ModelState.Executing;
            //         jpegData = texture2D.EncodeToJPG();
            //         break;
            //     case ModelState.Executing:
            //         compressedImageMsg rosImageData = new compressedImageMsg
            //         {
            //             format = "jpeg",
            //             data = jpegData,
            //         };
            //         this.modelState = ModelState.ReadOutput;
            //         // TODO
            //         // 2. get detection result
            //         // 3. show in unity
            //         rosConnection.Send("ar/compressed_image", rosImageData);
            //         break;
            //     case ModelState.ReadOutput:
            //         this.outputTensor = (TensorFloat)this.worker.PeekOutput();
            //         this.modelState = ModelState.Idle;
            //         this.outputTensor.AsyncReadbackRequest(_ => this.modelState = ModelState.PostProcessing);
            //         break;
            //     case ModelState.PostProcessing:
            //         this.outputTensor.MakeReadable();
            //         List<YoloItem> result =
            //             YoloModelOutputProcessor.ProcessModelOutput(this.outputTensor, this.threshold);
            //         this.yoloDebugOutput.ShowDebugInformation(this.inputTensor, result, this.cameraTransform);
            //         yoloRecognitionHandler.ShowRecognitions(result, this.cameraTransform);
            //         this.modelState = ModelState.PreProcessing;
            //         break;
            //     default:
            //         throw new ArgumentOutOfRangeException();
            // }
        }

        private void ShowCapturedTexture(Texture2D tex)
        {
            if (debugRawImage != null)
            {
                debugRawImage.texture = tex;
            }
        }

        private void Callback(SegmentationInfoMsg msg)
        {
            var segmentationImage = msg.segmentation;
            byte[] rawData = segmentationImage.data;

            int width = (int)segmentationImage.width;
            int height = (int)segmentationImage.height;

            Dictionary<int, BoundingBox2D> classToBBox = new Dictionary<int, BoundingBox2D>();

            // row-major: pixel (x, y) => rawData[y * width + x]
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    int classId = rawData[idx];

                    // 0 は背景 (または無効値) として無視
                    if (classId == 0)
                        continue;

                    if (!classToBBox.ContainsKey(classId))
                    {
                        classToBBox[classId] = new BoundingBox2D()
                        {
                            minX = x,
                            minY = y,
                            maxX = x,
                            maxY = y,
                        };
                    }
                    else
                    {
                        // 既存の BBox を更新
                        var box = classToBBox[classId];
                        box.minX = Mathf.Min(box.minX, x);
                        box.minY = Mathf.Min(box.minY, y);
                        box.maxX = Mathf.Max(box.maxX, x);
                        box.maxY = Mathf.Max(box.maxY, y);
                        classToBBox[classId] = box;
                    }
                }
            }

            // 3) 取得した bounding box 情報をもとに、YoloItem 的なリストを作る
            //    msg.detected_classes[classId-1] のように名前を取れる想定 (classId:1 => detected_classes[0])
            //    msg.scores[classId-1] でスコアを取れる想定
            List<YoloItem> deticItems = new List<YoloItem>();
            foreach (var kvp in classToBBox)
            {
                int classId = kvp.Key;
                var box = kvp.Value;

                // Detic から送られてくる classId が 1-based かどうか要確認
                int idxInMsg = classId - 1;
                if (idxInMsg < 0 || idxInMsg >= msg.detected_classes.Length)
                    continue;

                string className = msg.detected_classes[idxInMsg];
                float score = msg.scores[idxInMsg];

                // セグメンテーションの場合、(x, y) そのままだとトップレフト等の座標計算が
                // YOLO の座標系と異なることが多いので、必要に応じて正規化 (0〜1) するなど調整
                // ここでは単純に幅・高さからスケーリングする例

                float xMinNorm = box.minX / (float)width;
                float yMinNorm = box.minY / (float)height;
                float xMaxNorm = box.maxX / (float)width;
                float yMaxNorm = box.maxY / (float)height;

                // YoloItem 風の書き方(Version10 例)
                // ※ 実際の YoloItem 実装と座標系が合うように注意
                var newItem = YoloItem.FromVersion10(
                    topLeft: new Vector2(xMinNorm, yMinNorm),
                    bottomRight: new Vector2(xMaxNorm, yMaxNorm),
                    confidence: score,
                    classIndex: (int)classId, // ObjectClass 変換が必要な場合は適宜
                    className: className
                );

                // newItem.MostLikelyClassName = className;
                // ↑ YoloItem の実装に合わせてクラス名を保持できるようにする

                deticItems.Add(newItem);
            }

            // 4) 生成したリストをもとにデバッグ表示 / ハンドラー呼び出し
            //    カメラ画像と合成して 2D Overlay したいなら、YoloDebugOutput.ShowDebugInformation
            //    HoloLens の空間上に配置したい場合は、ワールド座標に変換して配置など
            yoloDebugOutput.ShowDebugInformation(null, deticItems, cameraTransform);
            yoloRecognitionHandler.ShowRecognitions(deticItems, cameraTransform);

        }

    /// <summary>
        ///     Method that is called when the object is destroyed.
        /// </summary>
        public void OnDestroy()
        {
            if (this.disposed)
            {
                return;
            }

            this.disposed = true;
            SettingsProvider.PropertyChanged -= this.SettingsProviderOnPropertyChanged;
            WebCamTextureAccess.Stop();
            this.inputTensor?.Dispose();
            this.worker?.Dispose();
        }

        private void SettingsProviderOnPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            switch (e.PropertyName)
            {
                case nameof(SettingsProvider.ModelExecutionOR):
                    this.UpdateModelPerformance();
                    break;
                case nameof(SettingsProvider.ThresholdOR):
                    this.UpdateThreshold();
                    break;
            }
        }

        private void UpdateModelPerformance()
        {
            this.layerCount = SettingsProvider.ModelExecutionOR switch
            {
                ModelExecutionMode.High => Parameters.LayersHigh,
                ModelExecutionMode.Low => Parameters.LayersLow,
                ModelExecutionMode.Full => int.MaxValue,
                _ => this.layerCount
            };
        }

        private void UpdateThreshold()
        {
            this.threshold = SettingsProvider.ThresholdOR switch
            {
                RecognitionThreshold.High => Parameters.ThresholdHigh,
                RecognitionThreshold.Medium => Parameters.ThresholdMedium,
                RecognitionThreshold.Low => Parameters.ThresholdLow,
                _ => this.threshold
            };
        }
    }
}