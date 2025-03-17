function busy()
    while true
        secs = rand(1:5)
        println("Sleeping for $secs seconds...")
        sleep(secs)
    end
end

busy()
