if __name__=='__main__':

    file=open("markerflow.txt","r")  #should throw error ffor multiple inputs etc
    output=open("out.txt","w")

    lines = file.readlines()

    line_out=" "
    tracknum=0
    for line in lines:
        if line!='\n':
            line_out+=line.translate(None,'[]\n')
            tracknum+=1
        if "]]]" in line: 
            output.write(line_out+"\n")
            line_out=" "
            print(tracknum)
            tracknum=0
        

    output.close()

