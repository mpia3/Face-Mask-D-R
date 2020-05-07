from google_images_download import google_images_download  # importing the library

response = google_images_download.googleimagesdownload()  # class instantiation

#our keywords: Andrew Garfield 2019, Angelina Jolie 2019, Anthony Hopkins 2019, Ben Affleck 2019, Beyonce Knowles 2019

arguments = {"keywords": "Andrew Garfield 2019, Angelina Jolie 2019, Anthony Hopkins 2019, Ben Affleck 2019, Beyonce Knowles 2019", "limit": 20,
             "print_urls": True}  # creating list of arguments
paths = response.download(arguments)  # passing the arguments to the function
print(paths)  # printing absolute paths of the downloaded images
