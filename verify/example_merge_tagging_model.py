from mtorch.caffenet import CaffeNet

# merged prototxt
net = CaffeNet('output/Tagging2K/Classification.prototxt')
# load merged prototxt with the old model weights (new weights will not be loaded)
net.load_weights('output/Tagging2K/Classification.caffemodel', ignore_shape_mismatch=False)
# text-only prototxt and model
textnet = CaffeNet('output/Tagging2K/Classification_text.prototxt')
# do not silently ignore shape mismatch for now, to make sure everythign is right
textnet.load_weights('output/Tagging2K/Classification_text.caffemodel', ignore_shape_mismatch=False)

# now update the
for lname, model in textnet.models.iteritems():
    lname = lname.replace('_taggingv2', '_text')
    tmodel = net.models[lname]
    if hasattr(model, 'weight'):
        print('{} weight'.format(lname))
        tmodel.weight = model.weight
    if hasattr(model, 'bias'):
        print('{} bias'.format(lname))
        tmodel.bias = model.bias

# save the merged model
net.save_weights('output/Tagging2K/Classification_new.caffemodel')
