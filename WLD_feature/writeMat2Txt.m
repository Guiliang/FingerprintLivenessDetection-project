function writeMat2Txt(matName, txtName)
%read the mat and save it to txt
stuct1=load(matName);
c_cell1 = struct2cell(stuct1);
data = cell2mat(c_cell1);
fileID = fopen(txtName,'w');

[m,n]=size(data);
 for i=1:1:m
    for j=1:1:n
       if j==n
         fprintf(fileID,'%d\n',data(i,j));
      else
        fprintf(fileID,'%d\t',data(i,j));
       end
    end
end
fclose(fileID);


end

