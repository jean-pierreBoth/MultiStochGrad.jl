module mnist



# big endian (network order) !!! must use ntoh(x)



export MnistData,
    getImages,
    getMeanImageLabel



struct MnistData
    images::Array{UInt8,3}
    labels::Array{UInt8,1}    
    #
    function MnistData(filenameImg, filenameLabel)
        imagesT = readidxImagefile(filenameImg)
        labelsT = readidxLabelsfile(filenameLabel)
        new(imagesT,labelsT)
    end
end


function readidxImagefile(filename::String)
    idfile = open(filename)
    # read 4 bytes (magic)
    val = read(idfile, UInt32)
    magic = Int32(ntoh(val))
    val = read(idfile, UInt32)
    nbitem = Int32(ntoh(val))
    val = read(idfile, UInt32)
    nbrow = Int32(ntoh(val))
    val = read(idfile, UInt32)
    nbcol = Int32(ntoh(val))
    images = Array{UInt8,3}(undef, nbrow,nbcol, nbitem)
    # then read matrix of bytes by row
    for k in 1:nbitem
        for i in 1:nbrow
            row = Vector{UInt8}(undef, nbcol)
            unsafe_read(idfile, pointer(row), nbcol*sizeof(UInt8))
            images[i,:,k] = row
        end
    end
    images
end




function readidxLabelsfile(filenameLabel)
   idfile = open(filenameLabel)
    # read 4 bytes (magic)
    val = read(idfile, UInt32)
    magic = Int32(ntoh(val))
    val = read(idfile, UInt32)
    nbitem = Int32(ntoh(val))
    #
    labels = Vector{UInt8}(undef, nbitem)
    unsafe_read(idfile, pointer(labels), sizeof(UInt8)*nbitem)
    labels
end


function getImages(mnist::MnistData , label::UInt8)
    idx=find(x-> x==label, mnist.labels)
    return mnist.images[:,:,idx]
end


function getMeanImageLabel(mnist::MnistData,  label::UInt8)
    idx=find(x-> x==label, mnist.labels)
    meanIL=zeros(Float64, mnist.images[:,:,1])
    meanIL = mean(mnist.images[:,:,idx],3)
    meanIL[:,:,1]
end



end

