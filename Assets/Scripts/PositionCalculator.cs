using UnityEngine;

namespace Assets.Scripts
{
    /// <summary>
    ///     Helper class for transforming detected positions in an image to world coordinate system coordinates.
    /// </summary>
    public class PositionCalculator
    {
        private static WebCamTextureAccess WebCamTextureAccess => WebCamTextureAccess.Instance;

        private const float RaycastMaxDistance = 5f;
        private static readonly LayerMask SpatialMeshLayerMask = 1 << 31;

        /// <summary>
        ///     Calculates the center or bottom point position of the yolo object in world space.
        /// </summary>
        /// <param name="yoloItem">Yolo item from the model.</param>
        /// <param name="cameraTransform">Current camera position.</param>
        /// <returns>Position in world space.</returns>
        public static Vector3? CalculatePointInSpace(YoloItem yoloItem, CameraTransform cameraTransform)
        {
            float centerX = yoloItem.Center.x;
            float centerY = yoloItem.Center.y;
            float flippedY = Screen.height - centerY;

            Vector3 screenPos = new Vector3(centerX, flippedY, 0f);

            Ray ray = Camera.main.ScreenPointToRay(screenPos);
            if (Physics.Raycast(ray, out RaycastHit hitInfo, RaycastMaxDistance, SpatialMeshLayerMask))
            {
                return hitInfo.point; // 実空間での衝突点
            }
            return null;
        }

        /// <summary>
        ///     Calculates the corner point positions of the yolo object in world space.
        /// </summary>
        /// <param name="yoloItem">Yolo item from the model.</param>
        /// <param name="cameraTransform">Current camera position.</param>
        /// <returns>The four corner points.</returns>
        public static Vector3[] CalculateCornerPoints(YoloItem yoloItem, CameraTransform cameraTransform)
        {
            Vector2 topRight = new Vector2(yoloItem.BottomRight.x, yoloItem.TopLeft.y);
            Vector2 bottomLeft = new Vector2(yoloItem.TopLeft.x, yoloItem.BottomRight.y);

            Vector2[] corners = new[]
            {
                yoloItem.TopLeft,
                topRight,
                yoloItem.BottomRight,
                bottomLeft
            };

            Vector3[] cornerPoints = new Vector3[4];

            for (int i = 0; i < 4; i++)
            {
                float x = corners[i].x;
                float y = corners[i].y;
                float flippedY = Screen.height - y;
                // スケーリングがいるならここで適用

                Ray ray = Camera.main.ScreenPointToRay(new Vector3(x, flippedY, 0f));
                if (Physics.Raycast(ray, out RaycastHit hit, RaycastMaxDistance, SpatialMeshLayerMask))
                {
                    cornerPoints[i] = hit.point;
                }
                else
                {
                    cornerPoints[i] = Vector3.zero; // or Vector3.negativeInfinity
                }
            }

            return cornerPoints;
        }

        private static Vector2 ScaleBack(Vector2 detectedPosition)
        {
            // ML model uses different resolution than the actual camera resolution => scale the detected position back to camera resolution.
            // After scaling the coordinates are in the range (-0.5, -0.5) - (0.5, 0.5) (=> center coordinate (0, 0)).
            int cameraResolutionX = WebCamTextureAccess.ActualCameraSize.x;
            int cameraResolutionY = WebCamTextureAccess.ActualCameraSize.y;
            return new Vector2(
                (detectedPosition.x / Parameters.ModelImageResolution.x * cameraResolutionX - (float)cameraResolutionX / 2) / cameraResolutionX,
                (detectedPosition.y / Parameters.ModelImageResolution.y * cameraResolutionY - (float)cameraResolutionY / 2) / cameraResolutionY
            );
        }

        /// <summary>
        ///     Converts a position inside the image to a position in space.
        /// </summary>
        /// <param name="cameraTransform">Current camera position.</param>
        /// <param name="positionInImage">Relative position inside the image (range (-0.5, -0.5) - (0.5, 0.5))</param>
        /// <returns>Position of the detected object one unit in front of the camera.</returns>
        public static Vector3 GetPositionInSpace(CameraTransform cameraTransform, Vector2 positionInImage)
        {
            // Move from the camera origin towards the position inside the image.
            // Image is placed one unit in front of the camera => add forward vector.
            // Size of the image is given by the virtualProjectionPlaneSize vector.
            return cameraTransform.Position + cameraTransform.Up * Parameters.HeightOffset + cameraTransform.Forward +
                   cameraTransform.Right * (positionInImage.x * Parameters.VirtualProjectionPlane.x) -
                   cameraTransform.Up * (positionInImage.y * Parameters.VirtualProjectionPlane.y);
        }

        private static Vector3? CastOnSpatialMap(Vector3 positionInSpace, CameraTransform cameraTransform)
        {
            // Try to send a cast to the (invisible) spatial mesh.
            // Cast origin is slightly above the camera position since the front camera of the HoloLens is placed above the eyes.
            Vector3 sphereCastOrigin = cameraTransform.Position + Parameters.SphereCastOffset * cameraTransform.Up;
            Vector3 direction = positionInSpace - sphereCastOrigin;

            if (PhysicsCaller.SphereCastOnSpatialMesh(sphereCastOrigin, direction, out RaycastHit hitInfo))
            {
                return hitInfo.point;
            }

            return null;
        }

        /// <summary>
        ///     Determines whether the object is visible in the current camera view.
        /// </summary>
        /// <param name="position">Position of the object.</param>
        /// <returns>Whether the object is visible in the current camera view.</returns>
        public static bool IsObjectInCameraView(Vector3 position)
        {
            Vector3 viewPos = Camera.main.WorldToViewportPoint(position);
            return (viewPos.x >= 0f && viewPos.x <= 1f) &&
                   (viewPos.y >= 0f && viewPos.y <= 1f) &&
                   viewPos.z >= 0f;
        }
    }
}