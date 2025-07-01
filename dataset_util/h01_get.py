

import cloudvolume
import google.colab.auth
google.colab.auth.authenticate_user()
c3_cloudvolume = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', progress=True)

mesh = c3_cloudvolume.mesh.get(3896803064)

# Mesh vertices are in nanometers (not voxels)
list(mesh.values())[0].vertices


# Request the skeleton for a single c3 segment id
skel = c3_cloudvolume.skeleton.get(3896803064)


# Skeleton vertices are in nanometers (not voxels)
skel.vertices