import numpy as np
import collections
import os
import shutil

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


if __name__ == "__main__":
    INPUT_DIR = "../data/faces/scan7"
    cameras=read_cameras_text(f"{INPUT_DIR}/colmap/cameras.txt")
    images=read_images_text(f"{INPUT_DIR}/colmap/images.txt")
    shutil.rmtree(f"{INPUT_DIR}/image", ignore_errors=True)
    shutil.rmtree(f"{INPUT_DIR}/mask", ignore_errors=True)
    os.makedirs(f"{INPUT_DIR}/image", exist_ok=True)
    os.makedirs(f"{INPUT_DIR}/mask", exist_ok=True)
    K = np.eye(3)
    K[0, 0] = cameras[1].params[0]
    K[1, 1] = cameras[1].params[1]
    K[0, 2] = cameras[1].params[2]
    K[1, 2] = cameras[1].params[3]

    cameras_npz_format = {}

    for idx, ii in enumerate(images.keys()):
        cur_image=images[ii]
        img_name = cur_image.name
        shutil.copy(f"{INPUT_DIR}/colmap/image/{img_name}", f"{INPUT_DIR}/image/{idx:03d}.png")
        shutil.copy(f"{INPUT_DIR}/colmap/mask/{img_name}.png", f"{INPUT_DIR}/mask/{idx:03d}.png")

        M=np.zeros((3,4))
        M[:,3]=cur_image.tvec
        M[:3,:3]=qvec2rotmat(cur_image.qvec)

        P=np.eye(4)
        P[:3,:] = K@M
        cameras_npz_format['world_mat_%d' % idx] = P

    np.savez(
            f"{INPUT_DIR}/cameras_before_normalization.npz",
            **cameras_npz_format)
