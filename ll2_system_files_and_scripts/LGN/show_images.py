import matplotlib.pyplot as plt

for image_name in ['8068', '108069', '130034', '163062', 'imk00895', 'imk01220', 'imk01261', 'imk01950', 'imk04208', 'pippin_Mex07_023']:

  image_f_name = 'natural_images/' + image_name + '.tiff'
  im = plt.imread(image_f_name)
  plt.imshow(im, cmap=plt.cm.Greys_r)
  plt.title('%s' % (image_name))
  plt.show()


