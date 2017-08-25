require 'rubygems'
require 'pdf-reader'


# File.open('raw_resumes.txt', 'w') do |fl|
#     for file in Dir['resumes/**/*.pdf']
#       reader = PDF::Reader.new(file)
#       puts reader.info
#       #puts reader.metadata
#         reader.pages.each do |page|
#             text_list = page.text.split("\n")
#             prev_line = nil
#             write_next = false
#             for line in text_list
#                 if write_next
#                     fl.write(line + "\n")
#                     prev_line = nil
#                     write_next = false
#                     next
#                 end
#                 if line =~ /\d/
#                     if !prev_line.nil?
#                         fl.write(prev_line + "\n")
#                     end
#                     fl.write(line + "\n")
#                     write_next = true
#                 end
#                 prev_line = line
#             end
#         end
#     end
# end





def pdf_to_text(file, noblank = true)
  spec = file.sub(/.pdf$/, '')
  `pdftotext #{spec}.pdf`
  # file = File.new("#{spec}.txt")
  # text = []
  # file.readlines.each do |l|
  #   l.chomp! if noblank
  #   if l.length &gt; 0
  #   text &lt;&lt; l
  #   end
  # end
  # file.close
  # text
end


File.open('raw_resumes.txt', 'w') do |fl|
  for file in Dir['resumes/**/*.txt']
    File.open(file, 'r') do |f|
      f.each_line do |line|
        line = line.encode('UTF-8', 'binary', :invalid => :replace, :undef => :replace)
        if line != "\n"
          fl.puts(line)
        end
      end
    end
  end
end




