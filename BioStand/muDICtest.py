import muDIC as dic

path = r'C:\Users\Lenovo\PycharmProjects\MainBioStand\BioStand\files\EcoflonPPE-N10-15x15\grayscale_image'
image_stack = dic.image_stack_from_folder(path,file_type=".tif")
mesher = dic.Mesher()
mesh = mesher.mesh(image_stack)
inputs = dic.DICInput(mesh, image_stack)
dic_job = dic.DICAnalysis(inputs)
results = dic_job.run()
fields = dic.Fields(results)
true_strain = fields.true_strain()
viz = dic.Visualizer(fields, images=image_stack)
for i in range(79):
    viz.show(field="True strain", component = (1,1), frame = 70)
